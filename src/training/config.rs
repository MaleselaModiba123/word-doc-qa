#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub max_seq_len: usize,
    pub checkpoint_every: usize,
    pub checkpoint_dir: String,
    pub log_every: usize,
}

impl TrainingConfig {
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