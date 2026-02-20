/// A single Question-Answer pair with its source context
#[derive(Debug, Clone)]
pub struct QAPair {
    pub question: String,
    pub context: String,
    pub answer: String,
}

/// Splits text into chunks of roughly `chunk_size` words with some overlap
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();

    if words.is_empty() {
        return chunks;
    }

    let mut start = 0;
    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(chunk);

        if end == words.len() {
            break;
        }
        // Move forward by chunk_size minus overlap
        start += chunk_size.saturating_sub(overlap);
    }

    chunks
}

/// Generates Q&A pairs from raw document text using a sliding window approach.
/// This creates training examples the model can learn from.
pub fn generate_qa_pairs(text: &str) -> Vec<QAPair> {
    let mut pairs = Vec::new();

    // Split into lines for sentence-level processing
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.len() > 20) // skip very short lines
        .collect();

    // --- Strategy 1: Date & event extraction ---
    // Look for lines that contain dates or event-like patterns
    for (i, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();

        // Look for date patterns (e.g. "January", "February", months)
        let months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ];

        for month in &months {
            if lower.contains(month) {
                // Build context from surrounding lines
                let context_start = i.saturating_sub(2);
                let context_end = (i + 3).min(lines.len());
                let context = lines[context_start..context_end].join(" ");

                // Generate a question about this date
                let question = format!("What event happens in {} mentioned in the document?", capitalize(month));
                pairs.push(QAPair {
                    question,
                    context: context.clone(),
                    answer: line.to_string(),
                });

                // Also add a "when" question
                let question2 = format!("When is the event described as '{}'?", truncate(line, 40));
                pairs.push(QAPair {
                    question: question2,
                    context,
                    answer: line.to_string(),
                });

                break; // only one pair per line per month match
            }
        }

        // Look for meeting/ceremony/graduation keywords
        let keywords = ["meeting", "ceremony", "graduation", "held", "session", "committee", "board", "hdc", "council"];
        for keyword in &keywords {
            if lower.contains(keyword) {
                let context_start = i.saturating_sub(2);
                let context_end = (i + 3).min(lines.len());
                let context = lines[context_start..context_end].join(" ");

                let question = format!("What does the document say about {}?", keyword);
                pairs.push(QAPair {
                    question,
                    context: context.clone(),
                    answer: line.to_string(),
                });
                break;
            }
        }
    }

    // --- Strategy 2: Sliding window chunks ---
    // Create general comprehension pairs from text chunks
    let chunks = chunk_text(text, 80, 20);
    for (i, chunk) in chunks.iter().enumerate() {
        // Create a "what does this section say" style pair
        let question = format!("What is described in section {} of the document?", i + 1);
        pairs.push(QAPair {
            question,
            context: chunk.clone(),
            answer: chunk.split('.').next().unwrap_or(chunk).trim().to_string(),
        });
    }

    // --- Strategy 3: Number/count questions ---
    // Look for lines with numbers that might answer "how many" questions
    for (i, line) in lines.iter().enumerate() {
        if line.chars().any(|c| c.is_numeric()) {
            let context_start = i.saturating_sub(2);
            let context_end = (i + 3).min(lines.len());
            let context = lines[context_start..context_end].join(" ");

            let question = "How many times or occurrences are mentioned in this section?".to_string();
            pairs.push(QAPair {
                question,
                context,
                answer: line.to_string(),
            });
        }
    }

    println!("Generated {} Q&A pairs from document", pairs.len());
    pairs
}

/// Capitalizes the first letter of a string
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Truncates a string to max_len characters, adding "..." if truncated
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_text(text, 4, 1);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_generate_qa_pairs() {
        let text = "The graduation ceremony will be held in November 2026. The HDC committee meeting was held in January.";
        let pairs = generate_qa_pairs(text);
        assert!(!pairs.is_empty());
    }
}