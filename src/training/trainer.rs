use std::time::Instant;
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        backend::{AutodiffBackend, Backend},
        ElementConversion, Int, Tensor,
    },
    data::dataloader::DataLoaderBuilder,
};
use crate::{
    data::{QABatcher, QADataset},
    model::{QATransformer, QATransformerConfig},
    training::{
        checkpoint::{save_checkpoint, save_checkpoint_metadata},
        config::TrainingConfig,
        metrics::{EpochMetrics, MetricsAccumulator, TrainingHistory},
    },
};

/// Computes cross-entropy loss between logits [batch, seq] and labels [batch]
fn compute_loss<B: AutodiffBackend>(
    logits: Tensor<B, 2>,
    labels: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    burn::tensor::activation::softmax(logits, 1)
        .log()
        .mul_scalar(-1.0f32)
        .select(1, labels)
        .mean()
}

/// Computes how many start position predictions match labels
fn compute_correct<B: Backend>(
    logits: Tensor<B, 2>,
    labels: Tensor<B, 1, Int>,
) -> usize {
    let [batch_size, _] = logits.dims();
    let predictions = logits.argmax(1).reshape([batch_size]);
    let eq = predictions.equal(labels);
    let sum: i32 = eq.int().sum().into_scalar().elem();
    sum as usize
}

/// Runs one training epoch
fn train_epoch<B: AutodiffBackend>(
    model: &mut QATransformer<B>,
    optimizer: &mut impl Optimizer<QATransformer<B>, B>,
    dataset: QADataset,
    config: &TrainingConfig,
    device: &B::Device,
) -> (f64, f64)
where
    B::InnerBackend: Backend,
{
    let batcher = QABatcher::<B>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(dataset);

    let mut accumulator = MetricsAccumulator::new();
    let mut batch_num = 0;

    for batch in dataloader.iter() {
        let batch_size = batch.labels.dims()[0];

        let (start_logits, end_logits) = model.forward_qa(batch.input_ids);

        let start_loss = compute_loss(start_logits.clone(), batch.labels.clone());
        let end_loss = compute_loss(end_logits, batch.labels.clone());
        let loss = (start_loss + end_loss) / 2.0f32;

        let loss_scalar: f32 = loss.clone().into_scalar().elem();
        let correct = compute_correct(start_logits.inner(), batch.labels.inner());
        accumulator.update(loss_scalar as f64, correct, batch_size);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, model);
        *model = optimizer.step(config.learning_rate, model.clone(), grads);

        batch_num += 1;
        if batch_num % config.log_every == 0 {
            println!(
                "  Batch {:>4} | Loss: {:.4} | Acc: {:.2}%",
                batch_num,
                accumulator.avg_loss(),
                accumulator.accuracy() * 100.0
            );
        }
    }

    (accumulator.avg_loss(), accumulator.accuracy())
}

/// Runs one validation epoch — no gradient updates
fn val_epoch<B: AutodiffBackend>(
    model: &QATransformer<B>,
    dataset: QADataset,
    config: &TrainingConfig,
    device: &B::Device,
) -> (f64, f64)
where
    B::InnerBackend: Backend,
{
    let batcher = QABatcher::<B::InnerBackend>::new(device.clone());
    let valid_model = model.valid();

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(dataset);

    let mut accumulator = MetricsAccumulator::new();

    for batch in dataloader.iter() {
        let batch_size = batch.labels.dims()[0];
        let (start_logits, _end_logits) = valid_model.forward_qa(batch.input_ids);

        let logits = start_logits.clone();
        let soft = burn::tensor::activation::softmax(logits, 1)
            .log()
            .mul_scalar(-1.0f32);
        let selected = soft.select(1, batch.labels.clone());
        let mean_scalar: f32 = selected.mean().into_scalar().elem();

        let correct = compute_correct(start_logits, batch.labels);
        accumulator.update(mean_scalar as f64, correct, batch_size);
    }

    (accumulator.avg_loss(), accumulator.accuracy())
}

/// Main training entry point
pub fn train<B: AutodiffBackend>(
    train_dataset: QADataset,
    val_dataset: QADataset,
    config: TrainingConfig,
    model_config: QATransformerConfig,
    device: B::Device,
) -> (QATransformer<B>, TrainingHistory)
where
    B::InnerBackend: Backend,
{
    println!("=== Starting Training ===");
    println!("Epochs:        {}", config.num_epochs);
    println!("Batch size:    {}", config.batch_size);
    println!("Learning rate: {}", config.learning_rate);
    println!("Train items:   {}", train_dataset.len());
    println!("Val items:     {}", val_dataset.len());
    println!();

    let mut model = model_config.init::<B>(&device);
    let mut optimizer = AdamConfig::new().init();
    let mut history = TrainingHistory::new();
    let mut best_val_loss = f64::INFINITY;

    for epoch in 1..=config.num_epochs {
        let epoch_start = Instant::now();
        println!("--- Epoch {} / {} ---", epoch, config.num_epochs);

        // Clone datasets for each epoch since DataLoader consumes them
        let train_data = QADataset::from_items(train_dataset.items.clone());
        let val_data = QADataset::from_items(val_dataset.items.clone());

        let (train_loss, train_acc) =
            train_epoch(&mut model, &mut optimizer, train_data, &config, &device);

        let (val_loss, val_acc) =
            val_epoch(&model, val_data, &config, &device);

        let duration = epoch_start.elapsed().as_secs_f64();

        let metrics = EpochMetrics {
            epoch,
            train_loss,
            val_loss,
            train_accuracy: train_acc,
            val_accuracy: val_acc,
            duration_secs: duration,
        };

        history.push(metrics);

        if epoch % config.checkpoint_every == 0 {
            if let Err(e) = save_checkpoint(&model, epoch, val_loss, &config.checkpoint_dir) {
                eprintln!("Warning: Could not save checkpoint: {}", e);
            }
            let _ = save_checkpoint_metadata(epoch, train_loss, val_loss, &config.checkpoint_dir);
        }

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            println!("  *** New best val loss: {:.4} ***", best_val_loss);
        }
    }

    println!("\n=== Training Complete ===");
    println!("Best val loss: {:.4}", history.best_val_loss());

    if let Err(e) = history.save_csv("training_history.csv") {
        eprintln!("Warning: Could not save history: {}", e);
    }

    (model, history)
}