use std::fs;
use std::path::Path;
use burn::{
    module::Module, record::{CompactRecorder, Recorder}, tensor::backend::Backend
};
use crate::model::QATransformer;

pub fn save_checkpoint<B: Backend>(
    model: &QATransformer<B>,
    epoch: usize,
    loss: f64,
    checkpoint_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(checkpoint_dir)?;
    let checkpoint_path = format!("{}/epoch_{:03}_loss_{:.4}", checkpoint_dir, epoch, loss);
    CompactRecorder::new()
        .record(model.clone().into_record(), Path::new(&checkpoint_path).into())
        .map_err(|e| format!("Failed to save checkpoint: {:?}", e))?;
    println!("Checkpoint saved: {}", checkpoint_path);
    Ok(())
}

pub fn save_checkpoint_metadata(
    epoch: usize,
    train_loss: f64,
    val_loss: f64,
    checkpoint_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let meta_path = format!("{}/epoch_{:03}_meta.txt", checkpoint_dir, epoch);
    let mut file = fs::File::create(&meta_path)?;
    writeln!(file, "epoch={}", epoch)?;
    writeln!(file, "train_loss={:.6}", train_loss)?;
    writeln!(file, "val_loss={:.6}", val_loss)?;
    Ok(())
}