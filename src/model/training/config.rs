/// All hyperparameters for training in one place
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of full passes through the training data
    pub num_epochs: usize,
    /// Number of samples per batch
    pub batch_size: usize,
    /// Learning rate for the optimizer
    pub learning_rate: f64,
    /// Weight decay for regularization
    pub weight_decay: f64,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Save a checkpoint every N epochs
    pub checkpoint_every: usize,
    /// Directory to save checkpoints
    pub checkpoint_dir: String,
    /// Print metrics every N batches
    pub log_every: usize,
}

impl TrainingConfig {
    /// Default configuration used for the main training run
    pub fn default() -> Self {
        Self {
            num_epochs: 10,
            batch_size: 8,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            max_seq_len: 256,
            checkpoint_every: 2,
            checkpoint_dir: "checkpoints".to_string(),
            log_every: 10,
        }
    }

    /// Smaller config for quick testing
    pub fn quick_test() -> Self {
        Self {
            num_epochs: 2,
            batch_size: 4,
            learning_rate: 1e-3,
            weight_decay: 0.01,
            max_seq_len: 256,
            checkpoint_every: 1,
            checkpoint_dir: "checkpoints".to_string(),
            log_every: 5,
        }
    }

    /// Larger config - second configuration for report comparison
    pub fn large() -> Self {
        Self {
            num_epochs: 15,
            batch_size: 16,
            learning_rate: 5e-5,
            weight_decay: 0.01,
            max_seq_len: 256,
            checkpoint_every: 3,
            checkpoint_dir: "checkpoints_large".to_string(),
            log_every: 10,
        }
    }
}