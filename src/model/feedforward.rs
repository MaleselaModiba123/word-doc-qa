use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Dropout, DropoutConfig, Gelu},
    tensor::{backend::Backend, Tensor},
};

/// Configuration for the feed-forward network
#[derive(Debug, Clone)]
pub struct FeedForwardConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub dropout: f64,
}

impl FeedForwardConfig {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size: hidden_size * 4, // standard transformer uses 4x expansion
            dropout: 0.1,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            fc1: LinearConfig::new(self.hidden_size, self.intermediate_size).init(device),
            fc2: LinearConfig::new(self.intermediate_size, self.hidden_size).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Gelu::new(),
        }
    }
}

/// Position-wise Feed-Forward Network
/// Applied after attention in each transformer layer
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
    activation: Gelu,
}

impl<B: Backend> FeedForward<B> {
    /// Forward: [batch, seq, hidden] -> [batch, seq, hidden]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);       // expand to intermediate size
        let x = self.activation.forward(x); // GELU activation
        let x = self.dropout.forward(x);    // dropout
        self.fc2.forward(x)                 // project back to hidden size
    }
}