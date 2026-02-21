use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor},
};
use crate::model::{
    attention::{MultiHeadAttention, MultiHeadAttentionConfig},
    feedforward::{FeedForward, FeedForwardConfig},
};

/// Configuration for one transformer encoder layer
#[derive(Debug, Clone)]
pub struct TransformerLayerConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub dropout: f64,
}

impl TransformerLayerConfig {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            dropout: 0.1,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerLayer<B> {
        TransformerLayer {
            attention: MultiHeadAttentionConfig::new(self.hidden_size, self.num_heads)
                .init(device),
            feed_forward: FeedForwardConfig::new(self.hidden_size).init(device),
            norm1: LayerNormConfig::new(self.hidden_size).init(device),
            norm2: LayerNormConfig::new(self.hidden_size).init(device),
        }
    }
}

/// One full Transformer Encoder Layer:
/// x -> LayerNorm -> MultiHeadAttention -> residual
///   -> LayerNorm -> FeedForward -> residual
#[derive(Module, Debug)]
pub struct TransformerLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    feed_forward: FeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
}

impl<B: Backend> TransformerLayer<B> {
    /// Forward pass with pre-norm residual connections
    /// x: [batch, seq_len, hidden] -> [batch, seq_len, hidden]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention block with residual connection (pre-norm style)
        let normed = self.norm1.forward(x.clone());
        let attn_out = self.attention.forward(normed);
        let x = x + attn_out; // residual

        // Feed-forward block with residual connection
        let normed = self.norm2.forward(x.clone());
        let ff_out = self.feed_forward.forward(normed);
        x + ff_out // residual
    }
}