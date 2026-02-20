mod data;

use data::{load_docx, QATokenizer, QADataset, split_dataset};

fn main() {
    println!("=== Word Doc Q&A System ===\n");

    // --- Step 1: Load documents ---
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
                // Use sample text for testing if no documents found
                combined_text.push_str(
                    "The End of Year Graduation Ceremony will be held on 14 November 2026. \
                     The HDC held their meetings 4 times in 2024, in January, April, July, and October. \
                     The committee convened on 15 March 2024 for the annual review meeting."
                );
            }
        }
    }

    println!("\nTotal text loaded: {} characters\n", combined_text.len());

    // --- Step 2: Create tokenizer ---
    let max_length = 256;
    println!("Initializing tokenizer (max_length={})...", max_length);
    let tokenizer = QATokenizer::new(max_length);
    println!("Vocabulary size: {}\n", tokenizer.vocab_size());

    // --- Step 3: Build dataset ---
    println!("Building dataset from text...");
    let dataset = QADataset::new(&combined_text, &tokenizer);
    println!("Total items: {}\n", dataset.len());

    // --- Step 4: Split into train/val ---
    println!("Splitting dataset (80/20)...");
    let (train_dataset, val_dataset) = split_dataset(dataset.items, 0.8);
    println!("Train: {} items", train_dataset.len());
    println!("Val:   {} items\n", val_dataset.len());

    println!("Data pipeline complete! Ready for model training.");
    println!("\nNext step: Run training with `cargo run -- train`");
}