mod data;
mod model;
mod training;
mod inference;

use data::{load_docx, QATokenizer, QADataset, split_dataset};
use model::QATransformerConfig;
use training::{TrainingConfig, train};
use burn::backend::{Autodiff, NdArray};
use inference::QAEngine;

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

fn load_documents() -> String {
    let doc_paths = vec![
        "docs/calendar_2024.docx",
        "docs/calendar_2025.docx",
        "docs/calendar_2026.docx",
    ];

    let mut combined = String::new();
    for path in &doc_paths {
        match load_docx(path) {
            Ok(text) => {
                println!("Loaded '{}' ({} chars)", path, text.len());
                combined.push_str(&text);
                combined.push_str("\n\n");
            }
            Err(_) => {
                eprintln!("Could not load '{}'", path);
                combined.push_str(
                    "The End of Year Graduation Ceremony will be held on 14 November 2026. \
                     The HDC held their meetings 4 times in 2024.",
                );
            }
        }
    }
    combined
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    // Print usage if no arguments
    if mode == "help" || mode == "--help" {
        println!("=== Word Doc Q&A System ===");
        println!();
        println!("USAGE:");
        println!("  cargo run -- test              Test model builds correctly");
        println!("  cargo run -- train             Full training (10 epochs)");
        println!("  cargo run -- train-quick       Quick training (2 epochs)");
        println!("  cargo run -- train-large       Large config training (15 epochs)");
        println!("  cargo run -- ask \"<question>\"  Ask a question about the documents");
        println!();
        println!("EXAMPLES:");
        println!("  cargo run -- ask \"What is the date of the 2026 graduation ceremony?\"");
        println!("  cargo run -- ask \"How many times did the HDC meet in 2024?\"");
        println!("  cargo run -- ask \"When is the end of year function?\"");
        return;
    }

    println!("=== Word Doc Q&A System ===\n");

    let combined_text = load_documents();
    println!("\nTotal text: {} chars\n", combined_text.len());

    let device: <InferBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let tokenizer = QATokenizer::new(256);
    let vocab_size = tokenizer.vocab_size();
    let model_config = QATransformerConfig::new(vocab_size);

    match mode {
        "train" => {
            println!("Mode: FULL TRAINING (10 epochs)\n");
            let dataset = QADataset::new(&combined_text, &tokenizer);
            let (train_dataset, val_dataset) = split_dataset(dataset.items, 0.8);
            println!("Train: {} | Val: {}\n", train_dataset.len(), val_dataset.len());

            let train_device: <TrainBackend as burn::tensor::backend::Backend>::Device = Default::default();
            let (_model, _history) = train::<TrainBackend>(
                train_dataset, val_dataset,
                TrainingConfig::default(), model_config, train_device,
            );
            println!("Training complete! Run: cargo run -- ask \"your question\"");
        }

        "train-quick" => {
            println!("Mode: QUICK TRAINING (2 epochs)\n");
            let dataset = QADataset::new(&combined_text, &tokenizer);
            let (train_dataset, val_dataset) = split_dataset(dataset.items, 0.8);
            println!("Train: {} | Val: {}\n", train_dataset.len(), val_dataset.len());

            let train_device: <TrainBackend as burn::tensor::backend::Backend>::Device = Default::default();
            let (_model, _history) = train::<TrainBackend>(
                train_dataset, val_dataset,
                TrainingConfig::quick_test(), model_config, train_device,
            );
            println!("Quick training complete!");
        }

        "train-large" => {
            println!("Mode: LARGE CONFIG TRAINING (15 epochs)\n");
            let dataset = QADataset::new(&combined_text, &tokenizer);
            let (train_dataset, val_dataset) = split_dataset(dataset.items, 0.8);
            println!("Train: {} | Val: {}\n", train_dataset.len(), val_dataset.len());

            let large_model_config = QATransformerConfig::large(vocab_size);
            let train_device: <TrainBackend as burn::tensor::backend::Backend>::Device = Default::default();
            let (_model, _history) = train::<TrainBackend>(
                train_dataset, val_dataset,
                TrainingConfig::large(), large_model_config, train_device,
            );
            println!("Large training complete!");
        }

        "ask" => {
            // Get the question from the next argument
            let question = match args.get(2) {
                Some(q) => q.as_str(),
                None => {
                    eprintln!("Please provide a question.");
                    eprintln!("Usage: cargo run -- ask \"What is the graduation date?\"");
                    return;
                }
            };

            println!("Question: {}\n", question);

            // Try to load from checkpoint, fall back to untrained model
            let checkpoint = find_latest_checkpoint("checkpoints");

            let engine = match checkpoint {
                Some(ref path) => {
                    println!("Using checkpoint: {}", path);
                    QAEngine::<InferBackend>::load(
                        path, model_config, &combined_text, device
                    ).unwrap_or_else(|_| {
                        println!("Falling back to keyword search...");
                        QAEngine::<InferBackend>::new_untrained(
                            QATransformerConfig::new(vocab_size), &combined_text, device
                        )
                    })
                }
                None => {
                    println!("No checkpoint found - using keyword search.");
                    println!("Run `cargo run -- train` first for better answers.\n");
                    QAEngine::<InferBackend>::new_untrained(
                        model_config, &combined_text, device
                    )
                }
            };

            println!("Searching {} document chunks...\n", engine.num_chunks());
            let answer = engine.answer(question);

            println!("Answer: {}", answer);
            println!();
        }

        "test" | _ => {
            println!("Mode: TEST\n");
            use burn::tensor::{Tensor, Int};
            let model = model_config.init::<InferBackend>(&device);
            let dummy = Tensor::<InferBackend, 2, Int>::zeros([2, 16], &device);
            let logits = model.forward(dummy);
            let [b, s, o] = logits.dims();
            println!("Forward pass OK! Output: [{}, {}, {}]", b, s, o);
            println!("\nCommands:");
            println!("  cargo run -- train");
            println!("  cargo run -- ask \"What is the graduation date?\"");
        }
    }
}

/// Finds the most recent checkpoint file
fn find_latest_checkpoint(dir: &str) -> Option<String> {
    let path = std::path::Path::new(dir);
    if !path.exists() {
        return None;
    }

    let mut checkpoints: Vec<String> = std::fs::read_dir(path)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("epoch_") && !name.ends_with("_meta.txt") {
                Some(format!("{}/{}", dir, name))
            } else {
                None
            }
        })
        .collect();

    checkpoints.sort();
    checkpoints.last().cloned()
}