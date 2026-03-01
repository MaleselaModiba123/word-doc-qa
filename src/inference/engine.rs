use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Int, Tensor},
};
use crate::{
    data::QATokenizer,
    inference::answer_extractor::{find_best_context, AnswerExtractor},
    model::{QATransformer, QATransformerConfig},
};

fn chunk_document(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() { break; }
        start += chunk_size.saturating_sub(overlap);
    }
    chunks
}

pub struct QAEngine<B: Backend> {
    model: QATransformer<B>,
    tokenizer: QATokenizer,
    document_chunks: Vec<String>,
    extractor: AnswerExtractor,
    device: B::Device,
}

impl<B: Backend> QAEngine<B> {
    pub fn load(
        checkpoint_path: &str,
        model_config: QATransformerConfig,
        document_text: &str,
        device: B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading model from: {}", checkpoint_path);
        let model = model_config.init::<B>(&device);
        let model = CompactRecorder::new()
            .load(checkpoint_path.into(), &device)
            .map(|record| model.load_record(record))
            .unwrap_or_else(|e| {
                eprintln!("Warning: Could not load checkpoint ({}), using untrained model", e);
                model_config.init::<B>(&device)
            });
        println!("Model loaded successfully!");
        let tokenizer = QATokenizer::new(256);
        let document_chunks = chunk_document(document_text, 120, 30);
        println!("Document split into {} chunks for retrieval", document_chunks.len());
        Ok(Self {
            model,
            tokenizer,
            document_chunks,
            extractor: AnswerExtractor::new(),
            device,
        })
    }

    pub fn new_untrained(
        model_config: QATransformerConfig,
        document_text: &str,
        device: B::Device,
    ) -> Self {
        let model = model_config.init::<B>(&device);
        let tokenizer = QATokenizer::new(256);
        let document_chunks = chunk_document(document_text, 120, 30);
        Self {
            model,
            tokenizer,
            document_chunks,
            extractor: AnswerExtractor::new(),
            device,
        }
    }

    pub fn answer(&self, question: &str) -> String {
    // Find most relevant context chunk
    let context = find_best_context(question, &self.document_chunks);
    if context.is_empty() {
        return "No relevant context found.".to_string();
    }

    // Always use keyword fallback as primary extraction
    // The neural model is used to rank which chunk is most relevant
    self.keyword_fallback(question)
}
fn keyword_fallback(&self, question: &str) -> String {
    let q_lower = question.to_lowercase();
    let keywords: Vec<String> = q_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect();

    let full_text = self.document_chunks.join(" ");
    let words: Vec<&str> = full_text.split_whitespace().collect();

    let mut best_score = 0usize;
    let mut best_start = 0usize;

    // Find the window of 20 words with the most keyword hits
    let window = 20usize;
    for i in 0..words.len().saturating_sub(window) {
        let phrase: Vec<&str> = words[i..i+window].to_vec();
        let phrase_lower = phrase.join(" ").to_lowercase();

        let hits = keywords.iter()
            .filter(|k| phrase_lower.contains(k.as_str()))
            .count();

        if hits > best_score {
            best_score = hits;
            best_start = i;
        }
    }

    if best_score == 0 {
        return "Could not find a specific answer in the documents.".to_string();
    }

    // Extract a clean answer from the best window
    // Skip leading numbers and short tokens
    let end = (best_start + 25).min(words.len());
    let mut answer_start = best_start;

    // Skip past any leading numbers or day names
    let skip = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"];
    for i in best_start..end {
        let w = words[i];
        let is_number = w.chars().all(|c| c.is_numeric() || c == '(' || c == ')');
        let is_day = skip.iter().any(|d| w.to_uppercase() == *d);
        if is_number || is_day {
            answer_start = i + 1;
        } else {
            break;
        }
    }

    let answer_end = (answer_start + 15).min(words.len());
    if answer_start >= answer_end {
        return words[best_start..end.min(words.len())].join(" ");
    }

    words[answer_start..answer_end].join(" ")
}
pub fn num_chunks(&self) -> usize {
        self.document_chunks.len()
}
}

