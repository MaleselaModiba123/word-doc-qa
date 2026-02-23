use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Int, Tensor},
};
use crate::model::{
    Embeddings::{TransformerEmbeddings, TransformerEmbeddingsConfig},
    transformer_layer::{TransformerLayer, TransformerLayerConfig},
};

/// Full configuration for the Q&A Transformer model
#[derive(Debug, Clone)]
pub struct QATransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,    // minimum 6 as required by assignment
    pub num_heads: usize,
    pub max_position: usize,
    pub dropout: f64,
    pub max_seq_len: usize,
}

impl QATransformerConfig {
    /// Default configuration - balanced for this assignment
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 256,   // embedding dimension
            num_layers: 6,      // exactly meets the minimum requirement
            num_heads: 8,       // 256 / 8 = 32 per head
            max_position: 512,
            dropout: 0.1,
            max_seq_len: 256,
        }
    }

    /// Larger config for comparison experiments in the report
    pub fn large(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 384,
            num_layers: 8,
            num_heads: 8,       // 384 / 8 = 48 per head
            max_position: 512,
            dropout: 0.1,
            max_seq_len: 256,
        }
    }

    /// Initializes the model on the given device
    pub fn init<B: Backend>(&self, device: &B::Device) -> QATransformer<B> {
        // Build embedding layer
        let mut emb_config = TransformerEmbeddingsConfig::new(self.vocab_size, self.hidden_size);
        emb_config.max_position = self.max_position;
        emb_config.dropout = self.dropout;
        let embeddings = emb_config.init(device);

        // Build transformer layers (minimum 6 as per assignment)
        let layers: Vec<TransformerLayer<B>> = (0..self.num_layers)
            .map(|_| {
                TransformerLayerConfig::new(self.hidden_size, self.num_heads).init(device)
            })
            .collect();

        // Final layer norm
        let final_norm = LayerNormConfig::new(self.hidden_size).init(device);

        // Output projection: hidden -> 2 (start and end position logits for span extraction)
        let output_projection = LinearConfig::new(self.hidden_size, 2).init(device);

        QATransformer {
            embeddings,
            layers,
            final_norm,
            output_projection,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
        }
    }

    /// Counts total trainable parameters
    pub fn count_parameters(&self) -> usize {
        // Embeddings
        let token_emb = self.vocab_size * self.hidden_size;
        let pos_emb = self.max_position * self.hidden_size;

        // Each transformer layer:
        // - 4 attention matrices (Q, K, V, O): 4 * hidden^2
        // - 2 layer norms: 2 * 2 * hidden
        // - FFN: hidden * 4*hidden + 4*hidden * hidden = 8 * hidden^2
        let per_layer = 4 * self.hidden_size * self.hidden_size
            + 8 * self.hidden_size * self.hidden_size
            + 4 * self.hidden_size;
        let all_layers = self.num_layers * per_layer;

        // Final norm + output projection
        let final_layers = 2 * self.hidden_size + self.hidden_size * 2;

        token_emb + pos_emb + all_layers + final_layers
    }
}

/// The full Q&A Transformer model
/// Takes tokenized input and outputs start/end position logits
#[derive(Module, Debug)]
pub struct QATransformer<B: Backend> {
    embeddings: TransformerEmbeddings<B>,
    layers: Vec<TransformerLayer<B>>,
    final_norm: LayerNorm<B>,
    output_projection: Linear<B>,
    hidden_size: usize,
    num_layers: usize,
}

impl<B: Backend> QATransformer<B> {
    /// Forward pass
    /// input_ids: [batch, seq_len] -> logits: [batch, seq_len, 2]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Step 1: Embed tokens + positions
        let mut x = self.embeddings.forward(input_ids);

        // Step 2: Pass through all transformer layers
        for layer in &self.layers {
            x = layer.forward(x);
        }

        // Step 3: Final layer normalization
        let x = self.final_norm.forward(x);

        // Step 4: Project to output logits [batch, seq_len, 2]
        // The 2 outputs are: start position logit, end position logit
        self.output_projection.forward(x)
    }

    /// Returns start and end position logits separately
    /// Useful for computing loss and for inference
    pub fn forward_qa(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let logits = self.forward(input_ids); // [batch, seq_len, 2]
        let [batch, seq_len, _] = logits.dims();

        // Split into start and end logits
        let start_logits = logits
            .clone()
            .slice([0..batch, 0..seq_len, 0..1])
            .flatten::<2>(1,2); // [batch, seq_len]

        let end_logits = logits
            .slice([0..batch, 0..seq_len, 1..2])
            .flatten::<2>(1,2); // [batch, seq_len]

        (start_logits, end_logits)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}