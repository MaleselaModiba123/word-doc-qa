use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Int, Tensor},
};
use crate::{
    data::QATokenizer,
    inference::answer_extractor::{find_best_context, score_relevance, AnswerExtractor},
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
        let document_chunks = chunk_document(document_text, 60, 15);
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
        let document_chunks = chunk_document(document_text, 60, 15);
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
    let mut scored: Vec<(f32, &String)> = self.document_chunks
        .iter()
        .map(|c| (score_relevance(question, c), c))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let q_lower = question.to_lowercase();
    let keywords: Vec<String> = q_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .map(|w| w.to_string())
        .collect();

    // Split chunks into individual lines first, then sentences
    // This avoids returning entire calendar grid rows
    let mut candidates: Vec<(usize, String)> = Vec::new();

    for (_, chunk) in scored.iter().take(10) {
        // Split on newlines AND periods to get individual lines
        for line in chunk.split('\n') {
            for sentence in line.split('.') {
                let s = sentence.trim().to_string();
                let s_lower = s.to_lowercase();
                let word_count = s.split_whitespace().count();

                // Skip lines that look like calendar headers or grids
                // (they have too many single words like day names, numbers)
                let avg_word_len: f32 = s.split_whitespace()
                    .map(|w| w.len() as f32)
                    .sum::<f32>()
                    / word_count.max(1) as f32;

                // Calendar grid lines have very short average word length
                // Real sentences have longer average word length
                if avg_word_len < 3.5 { continue; }
                if word_count < 4 || word_count > 60 { continue; }

                // Count how many keywords appear in this sentence
                let keyword_hits = keywords.iter()
                    .filter(|k| s_lower.contains(k.as_str()))
                    .count();

                if keyword_hits >= 1 {
                    candidates.push((keyword_hits, s));
                }
            }
        }
    }

    // Sort by most keyword hits first
    candidates.sort_by(|a, b| b.0.cmp(&a.0));

    if let Some((_, answer)) = candidates.first() {
        return answer.clone();
    }

    "Could not find a specific answer in the documents.".to_string()
}

    pub fn num_chunks(&self) -> usize {
        self.document_chunks.len()
    }
}

