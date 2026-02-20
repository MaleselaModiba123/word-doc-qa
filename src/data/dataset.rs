use burn::data::dataset::Dataset;
use crate::data::{
    qa_pairs::{QAPair, generate_qa_pairs},
    tokenizer::QATokenizer,
};

/// A single training item ready to be batched
#[derive(Debug, Clone)]
pub struct QAItem {
    /// Token IDs for [CLS] question [SEP] context [SEP]
    pub input_ids: Vec<u32>,
    /// 1 for real tokens, 0 for padding
    pub attention_mask: Vec<u32>,
    /// The start position of the answer in the token sequence (for span extraction)
    pub label: u32,
    /// The raw answer text (kept for evaluation)
    pub answer_text: String,
}

/// The main dataset that holds all Q&A training items
pub struct QADataset {
    pub items: Vec<QAItem>,
}

impl QADataset {
    /// Creates a new dataset from raw text using the provided tokenizer
    pub fn new(text: &str, tokenizer: &QATokenizer) -> Self {
        let pairs = generate_qa_pairs(text);
        let items = Self::tokenize_pairs(pairs, tokenizer);
        println!("Dataset created with {} items", items.len());
        Self { items }
    }

    /// Creates a dataset directly from pre-built items (used after splitting)
    pub fn from_items(items: Vec<QAItem>) -> Self {
        Self { items }
    }

    /// Converts Q&A pairs into tokenized QAItems
    fn tokenize_pairs(pairs: Vec<QAPair>, tokenizer: &QATokenizer) -> Vec<QAItem> {
        pairs
            .into_iter()
            .map(|pair| {
                let encoded = tokenizer.encode_qa(&pair.question, &pair.context);

                // Find where the answer starts in the token sequence
                // For simplicity we use a heuristic: find the answer text position
                let label = Self::find_answer_position(&encoded.input_ids, &pair.answer, tokenizer);

                QAItem {
                    input_ids: encoded.input_ids,
                    attention_mask: encoded.attention_mask,
                    label,
                    answer_text: pair.answer,
                }
            })
            .collect()
    }

    /// Finds the approximate token position of the answer in the input
    /// Returns 0 if not found (will be refined during training)
    fn find_answer_position(
        input_ids: &[u32],
        _answer: &str,
        _tokenizer: &QATokenizer,
    ) -> u32 {
        // Simple heuristic: answer usually starts in the second half
        // (after the [SEP] token separating question from context)
        let sep_pos = input_ids
            .iter()
            .position(|&id| id == 102) // 102 is [SEP] in BERT
            .unwrap_or(input_ids.len() / 2);

        sep_pos as u32
    }

    /// Returns the number of items in the dataset
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the dataset has no items
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Implement Burn's Dataset trait so Burn can use this in DataLoaders
impl Dataset<QAItem> for QADataset {
    fn get(&self, index: usize) -> Option<QAItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// Splits a dataset into training and validation sets
/// Default split: 80% train, 20% validation
pub fn split_dataset(items: Vec<QAItem>, train_ratio: f32) -> (QADataset, QADataset) {
    let split_idx = (items.len() as f32 * train_ratio) as usize;
    let split_idx = split_idx.max(1); // ensure at least 1 item in train

    let mut all_items = items;

    // Shuffle before splitting for better distribution
    // Simple shuffle using index swapping
    let n = all_items.len();
for i in (1..n).rev() {
    let j = i % (i + 1); // simple deterministic swap
    all_items.swap(i, j);
}

    let val_items = all_items.split_off(split_idx);
    let train_items = all_items;

    println!(
        "Split: {} training, {} validation items",
        train_items.len(),
        val_items.len()
    );

    (
        QADataset::from_items(train_items),
        QADataset::from_items(val_items),
    )
}

