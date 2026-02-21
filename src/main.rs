mod data;
mod model;
mod training;

use data::{load_docx, QATokenizer, QADataset, split_dataset};
use model::QATransformerConfig;
use training::{TrainingConfig, train};
use burn::backend::{Autodiff, NdArray};

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("test");

    println!("=== Word Doc Q&A System ===\n");

    let doc_paths = vec![
        "docs/calendar_2024.docx",
        "docs/calendar_2025.docx",
        "docs/calendar_2026.docx",
    ];

    let mut combined_text = String::new();
    for path in &doc_paths {
        match load_docx(path) {
            Ok(text) => {
                println!("Loaded '{}' ({} chars)", path, text.len());
                combined_text.push_str(&text);
                combined_text.push_str("\n\n");
            }
            Err(e) => {
                eprintln!("Could not load '{}': {}", path, e);
                combined_text.push_str(
                    "The End of Year Graduation Ceremony will be held on 14 November 2026. \
                     The HDC held their meetings 4 times in 2024.",
                );
            }
        }
    }

    println!("\nTotal text: {} chars\n", combined_text.len());

    let tokenizer = QATokenizer::new(256);
    let vocab_size = tokenizer.vocab_size();
    let dataset = QADataset::new(&combined_text, &tokenizer);
    let (train_dataset, val_dataset) = split_dataset(dataset.items, 0.8);
    println!("Train: {} | Val: {}\n", train_dataset.len(), val_dataset.len());

    let device = Default::default();
    let model_config = QATransformerConfig::new(vocab_size);

    match mode {
        "train" => {
            println!("Mode: TRAINING\n");
            let train_config = TrainingConfig::default();
            let (_model, _history) = train::<TrainBackend>(
                train_dataset,
                val_dataset,
                train_config,
                model_config,
                device,
            );
            println!("\nTraining complete! History saved to training_history.csv");
        }
        "train-quick" => {
            println!("Mode: QUICK TRAINING (2 epochs)\n");
            let train_config = TrainingConfig::quick_test();
            let (_model, _history) = train::<TrainBackend>(
                train_dataset,
                val_dataset,
                train_config,
                model_config,
                device,
            );
            println!("\nQuick training complete!");
        }
        _ => {
            println!("Mode: TEST\n");
            use burn::tensor::{Tensor, Int};
            let model = model_config.init::<InferBackend>(&device);
            let dummy = Tensor::<InferBackend, 2, Int>::zeros([2, 16], &device);
            let logits = model.forward(dummy);
            let [b, s, o] = logits.dims();
            println!("Forward pass OK! Output: [{}, {}, {}]", b, s, o);
            println!("\nRun `cargo run -- train-quick` to test training.");
            println!("Run `cargo run -- train` for full training.");
        }
    }
}