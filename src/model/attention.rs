use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Dropout, DropoutConfig},
    tensor::{backend::Backend, Tensor},
};

/// Configuration for multi-head attention
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub dropout: f64,
}

impl MultiHeadAttentionConfig {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            dropout: 0.1,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        assert!(
            self.hidden_size % self.num_heads == 0,
            "hidden_size must be divisible by num_heads"
        );

        let head_dim = self.hidden_size / self.num_heads;

        MultiHeadAttention {
            query: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            key: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            value: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            out_proj: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            num_heads: self.num_heads,
            head_dim,
            hidden_size: self.hidden_size,
        }
    }
}

/// Multi-Head Self Attention
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    out_proj: Linear<B>,
    dropout: Dropout,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Forward pass
    /// x: [batch, seq_len, hidden] -> output: [batch, seq_len, hidden]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden] = x.dims();

        // Project to Q, K, V
        let q = self.query.forward(x.clone()); // [batch, seq, hidden]
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        // Reshape to [batch, heads, seq, head_dim]
        let q = self.split_heads(q, batch_size, seq_len);
        let k = self.split_heads(k, batch_size, seq_len);
        let v = self.split_heads(v, batch_size, seq_len);

        // Scaled dot-product attention
        // scores = Q * K^T / sqrt(head_dim)
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.swap_dims(2, 3); // [batch, heads, head_dim, seq]
        let scores = q.matmul(k_t) / scale; // [batch, heads, seq, seq]

        // Softmax over last dim
        let attn_weights = burn::tensor::activation::softmax(scores, 3);
        let attn_weights = self.dropout.forward(attn_weights);

        // Apply attention to values
        let context = attn_weights.matmul(v); // [batch, heads, seq, head_dim]

        // Merge heads back: [batch, seq, hidden]
        let context = context
            .swap_dims(1, 2)                                    // [batch, seq, heads, head_dim]
            .reshape([batch_size, seq_len, self.hidden_size]);  // [batch, seq, hidden]

        // Final projection
        self.out_proj.forward(context)
    }

    /// Splits hidden dim into multiple heads
    /// [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    fn split_heads(&self, x: Tensor<B, 3>, batch_size: usize, seq_len: usize) -> Tensor<B, 4> {
        x.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2) // [batch, heads, seq, head_dim]
    }
}