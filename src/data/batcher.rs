use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Int, Tensor},
};
use crate::data::dataset::QAItem;

#[derive(Debug, Clone)]
pub struct QABatch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub attention_mask: Tensor<B, 2, Int>,
    pub labels: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct QABatcher<B: Backend> {
    pub device: B::Device,
}

impl<B: Backend> QABatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, QAItem, QABatch<B>> for QABatcher<B> {
    fn batch(&self, items: Vec<QAItem>, device: &B::Device) -> QABatch<B> {
        let batch_size = items.len();
        let seq_len = items[0].input_ids.len();

        let input_ids_flat: Vec<i32> = items
            .iter()
            .flat_map(|item| item.input_ids.iter().map(|&id| id as i32))
            .collect();

        let attention_mask_flat: Vec<i32> = items
            .iter()
            .flat_map(|item| item.attention_mask.iter().map(|&m| m as i32))
            .collect();

        let labels_flat: Vec<i32> = items
            .iter()
            .map(|item| item.label as i32)
            .collect();

        let input_ids = Tensor::<B, 1, Int>::from_ints(
            input_ids_flat.as_slice(),
            device,
        )
        .reshape([batch_size, seq_len]);

        let attention_mask = Tensor::<B, 1, Int>::from_ints(
            attention_mask_flat.as_slice(),
            device,
        )
        .reshape([batch_size, seq_len]);

        let labels = Tensor::<B, 1, Int>::from_ints(
            labels_flat.as_slice(),
            device,
        );

        QABatch {
            input_ids,
            attention_mask,
            labels,
        }
    }
}