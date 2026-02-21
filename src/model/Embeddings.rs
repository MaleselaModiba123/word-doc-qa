use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Dropout, DropoutConfig},
    tensor::{backend::Backend, Int, Tensor},
};

/// Configuration for the embedding layer
#[derive(Debug, Clone)]
pub struct TransformerEmbeddingsConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub max_position: usize,
    pub dropout: f64,
}

impl TransformerEmbeddingsConfig {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size,
            max_position: 512,
            dropout: 0.1,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerEmbeddings<B> {
        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.hidden_size)
            .init(device);
        let position_embedding = EmbeddingConfig::new(self.max_position, self.hidden_size)
            .init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        TransformerEmbeddings {
            token_embedding,
            position_embedding,
            dropout,
            hidden_size: self.hidden_size,
        }
    }
}

/// Combines token embeddings + positional embeddings
#[derive(Module, Debug)]
pub struct TransformerEmbeddings<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    dropout: Dropout,
    hidden_size: usize,
}

impl<B: Backend> TransformerEmbeddings<B> {
    /// Forward pass: input_ids [batch, seq_len] -> embeddings [batch, seq_len, hidden]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Token embeddings: [batch, seq_len, hidden]
        let token_emb = self.token_embedding.forward(input_ids);

        // Position IDs: [1, seq_len] -> [batch, seq_len]
        let position_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .expand([batch_size, seq_len]);

        // Position embeddings: [batch, seq_len, hidden]
        let pos_emb = self.position_embedding.forward(position_ids);

        // Add token + positional embeddings, apply dropout
        let embeddings = token_emb + pos_emb;
        self.dropout.forward(embeddings)
    }
}