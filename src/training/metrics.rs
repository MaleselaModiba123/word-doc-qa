/// Tracks metrics for one epoch
#[derive(Debug, Clone, Default)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_accuracy: f64,
    pub val_accuracy: f64,
    pub duration_secs: f64,
}

impl EpochMetrics {
    pub fn print(&self) {
        println!(
            "Epoch {:>3} | Train Loss: {:.4} | Val Loss: {:.4} | Train Acc: {:.2}% | Val Acc: {:.2}% | Time: {:.1}s",
            self.epoch,
            self.train_loss,
            self.val_loss,
            self.train_accuracy * 100.0,
            self.val_accuracy * 100.0,
            self.duration_secs,
        );
    }
}

#[derive(Debug, Default)]
pub struct MetricsAccumulator {
    total_loss: f64,
    correct: usize,
    total: usize,
    num_batches: usize,
}

impl MetricsAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, loss: f64, correct: usize, total: usize) {
        self.total_loss += loss;
        self.correct += correct;
        self.total += total;
        self.num_batches += 1;
    }

    pub fn avg_loss(&self) -> f64 {
        if self.num_batches == 0 { return 0.0; }
        self.total_loss / self.num_batches as f64
    }

    pub fn accuracy(&self) -> f64 {
        if self.total == 0 { return 0.0; }
        self.correct as f64 / self.total as f64
    }
}

#[derive(Debug, Default)]
pub struct TrainingHistory {
    pub epochs: Vec<EpochMetrics>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, metrics: EpochMetrics) {
        metrics.print();
        self.epochs.push(metrics);
    }

    pub fn save_csv(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "epoch,train_loss,val_loss,train_accuracy,val_accuracy,duration_secs")?;
        for m in &self.epochs {
            writeln!(file, "{},{:.6},{:.6},{:.6},{:.6},{:.2}",
                m.epoch, m.train_loss, m.val_loss,
                m.train_accuracy, m.val_accuracy, m.duration_secs)?;
        }
        println!("Training history saved to {}", path);
        Ok(())
    }

    pub fn best_val_loss(&self) -> f64 {
        self.epochs.iter().map(|m| m.val_loss).fold(f64::INFINITY, f64::min)
    }
}