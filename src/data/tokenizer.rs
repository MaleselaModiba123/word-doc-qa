use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Wraps the HuggingFace tokenizer for our Q&A system
pub struct QATokenizer {
    tokenizer: Tokenizer,
    pub max_length: usize,
    pub vocab_size: usize,
    pub pad_token_id: u32,
    pub sep_token_id: u32,
    pub cls_token_id: u32,
}

impl QATokenizer {
    /// Creates a new tokenizer using a built-in WordPiece vocabulary.
    /// We use bert-base-uncased tokenizer as a base since it's well suited for Q&A.
    pub fn new(max_length: usize) -> Self {
        // Load a pretrained BERT tokenizer from a local file or build a basic one
        // We'll build a simple whitespace tokenizer with a basic vocab if BERT isn't available
        let tokenizer = Self::build_tokenizer(max_length);
        let vocab_size = tokenizer.get_vocab_size(true);

        // Standard BERT special token IDs
        let pad_token_id = tokenizer
            .token_to_id("[PAD]")
            .unwrap_or(0);
        let sep_token_id = tokenizer
            .token_to_id("[SEP]")
            .unwrap_or(102);
        let cls_token_id = tokenizer
            .token_to_id("[CLS]")
            .unwrap_or(101);

        QATokenizer {
            tokenizer,
            max_length,
            vocab_size,
            pad_token_id,
            sep_token_id,
            cls_token_id,
        }
    }

    /// Builds the tokenizer - tries to load BERT, falls back to whitespace tokenizer
    fn build_tokenizer(max_length: usize) -> Tokenizer {
        // Try loading from local file first (user can download bert tokenizer)
        if std::path::Path::new("tokenizer.json").exists() {
            match Tokenizer::from_file("tokenizer.json") {
                Ok(mut tok) => {
                    Self::configure_tokenizer(&mut tok, max_length);
                    println!("Loaded tokenizer from tokenizer.json");
                    return tok;
                }
                Err(e) => {
                    eprintln!("Could not load tokenizer.json: {}, building basic tokenizer", e);
                }
            }
        }

        // Build a basic BPE tokenizer from scratch as fallback
        Self::build_basic_tokenizer(max_length)
    }

    /// Configures padding and truncation on the tokenizer
    fn configure_tokenizer(tokenizer: &mut Tokenizer, max_length: usize) {
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            ..Default::default()
        }));

        tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        })).unwrap();
    }

    /// Builds a simple whitespace tokenizer with character-level fallback
    fn build_basic_tokenizer(max_length: usize) -> Tokenizer {
        use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::normalizers::Lowercase;
        use tokenizers::processors::template::TemplateProcessing;
        use tokenizers::decoders::bpe::BPEDecoder;
        use tokenizers::TokenizerBuilder;

        // Build a minimal tokenizer
        let mut tokenizer: Tokenizer = TokenizerBuilder::new()
            .with_model(BPE::default())
            .with_normalizer(Some(Lowercase))
            .with_pre_tokenizer(Some(Whitespace {}))
            .with_post_processor(Some(TemplateProcessing::default()))
            .with_decoder(Some(BPEDecoder::default()))
            .build()
            .unwrap_or_else(|_| {
                // Absolute fallback - build another basic tokenizer
                TokenizerBuilder::new()
                    .with_model(BPE::default())
                    .with_normalizer(Some(Lowercase))
                    .with_pre_tokenizer(Some(Whitespace {}))
                    .with_post_processor(Some(TemplateProcessing::default()))
                    .build()
                    .expect("Failed to build fallback tokenizer")
            })
            .into();

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(max_length),
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            ..Default::default()
        }));

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .unwrap();

        println!("Built basic whitespace tokenizer");
        tokenizer
    }

    /// Encodes a question + context pair into token IDs
    /// Format: [CLS] question [SEP] context [SEP]
    pub fn encode_qa(&self, question: &str, context: &str) -> TokenizedInput {
        let combined = format!("{} [SEP] {}", question, context);

        match self.tokenizer.encode(combined, true) {
            Ok(encoding) => {
                let mut ids: Vec<u32> = encoding.get_ids().to_vec();
                let mut mask: Vec<u32> = encoding.get_attention_mask().to_vec();

                // Pad or truncate to max_length
                Self::pad_or_truncate(&mut ids, self.max_length, self.pad_token_id);
                Self::pad_or_truncate(&mut mask, self.max_length, 0);

                TokenizedInput {
                    input_ids: ids,
                    attention_mask: mask,
                }
            }
            Err(_) => {
                // Fallback: simple character-based encoding
                self.simple_encode(&format!("{} {} ", question, context))
            }
        }
    }

    /// Simple fallback encoding using character codes
    fn simple_encode(&self, text: &str) -> TokenizedInput {
        let mut ids: Vec<u32> = text
            .chars()
            .take(self.max_length)
            .map(|c| (c as u32) % 30000)
            .collect();

        let mut mask: Vec<u32> = vec![1u32; ids.len()];

        Self::pad_or_truncate(&mut ids, self.max_length, self.pad_token_id);
        Self::pad_or_truncate(&mut mask, self.max_length, 0);

        TokenizedInput {
            input_ids: ids,
            attention_mask: mask,
        }
    }

    /// Pads or truncates a vector to the target length
    fn pad_or_truncate(vec: &mut Vec<u32>, target: usize, pad_value: u32) {
        if vec.len() < target {
            vec.resize(target, pad_value);
        } else {
            vec.truncate(target);
        }
    }

    /// Returns the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size.max(30522) // minimum BERT vocab size
    }
}

/// The output of tokenization - ready to be converted to tensors
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
}

