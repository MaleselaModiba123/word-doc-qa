/// Finds the best answer from the document text given start/end position scores
pub struct AnswerExtractor {
    pub max_answer_len: usize,
}

impl AnswerExtractor {
    pub fn new() -> Self {
        Self { max_answer_len: 50 }
    }

    /// Given start and end logits, find the best answer span positions
    /// Returns (start_idx, end_idx)
    pub fn extract_span(&self, start_logits: &[f32], end_logits: &[f32]) -> (usize, usize) {
        let seq_len = start_logits.len();
        let mut best_score = f32::NEG_INFINITY;
        let mut best_start = 0;
        let mut best_end = 0;

        for start in 0..seq_len {
            for end in start..seq_len.min(start + self.max_answer_len) {
                let score = start_logits[start] + end_logits[end];
                if score > best_score {
                    best_score = score;
                    best_start = start;
                    best_end = end;
                }
            }
        }

        (best_start, best_end)
    }

    /// Converts token positions back to a text answer using the context
    /// This uses a word-level approach since we have a simple tokenizer
    pub fn tokens_to_answer(
        &self,
        context: &str,
        start_pos: usize,
        end_pos: usize,
    ) -> String {
        let words: Vec<&str> = context.split_whitespace().collect();
        if words.is_empty() || start_pos >= words.len() {
            return "Could not extract answer.".to_string();
        }

        let end = end_pos.min(words.len() - 1);
        let start = start_pos.min(end);

        words[start..=end].join(" ")
    }
}

/// Scores a context chunk for relevance to a question
/// Returns a relevance score (higher = more relevant)
pub fn score_relevance(question: &str, context: &str) -> f32 {
    let q_words: std::collections::HashSet<&str> = question
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 3) // skip short words like "the", "is", etc.
        .collect();

    let c_words: Vec<&str> = context.split_whitespace().collect();

    let matches = c_words
        .iter()
        .filter(|w| {
            let clean = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            q_words.iter().any(|q| q.to_lowercase() == clean)
        })
        .count();

    matches as f32 / q_words.len().max(1) as f32
}

/// Finds the most relevant chunk of text for a question
pub fn find_best_context<'a>(question: &str, chunks: &'a [String]) -> &'a str {
    chunks
        .iter()
        .max_by(|a, b| {
            let score_a = score_relevance(question, a);
            let score_b = score_relevance(question, b);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.as_str())
        .unwrap_or("")
}