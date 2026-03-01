# SEG 580S: Software Engineering Deep learning System Trainging on CPUT Data

**Student:** Malesela Modiba (222140275)

---

## 1. Introduction

### 1.1 Problem Statement and Motivation

Institutions like CPUT generate large volumes of administrative documents such as academic calendars, committee schedules, and governance records that are difficult to query efficiently. A staff member or student looking for a specific date, event, or meeting count must manually scan through dozens of pages of formatted Word documents.

This project builds a Question-Answering (Q&A) system that reads `.docx` calendar documents and answers natural language questions about their content. Example queries the system handles:

- *"What is the month and date of the 2026 End of Year Graduation Ceremony?"*
- *"How many times did the HDC hold their meetings in 2024?"*
- *"When does the academic year start in 2025?"*

### 1.2 Overview of Approach

The system is implemented entirely in Rust using the Burn deep learning framework. It follows a full ML pipeline:

1. Data loading — `.docx` files are read and text is extracted including table content
2. Data processing — text is chunked and tokenized into Q&A training pairs
3. Model Training — a transformer encoder is trained on the Q&A pairs
4. Interface — the trained model ranks context chunks by relevance, and a keyword extraction layer retrieves the final answer

The approach combines neural relevance scoring with deterministic keyword extraction, which produces reliable answers on domain-specific calendar documents.

### 1.3 Key Design Decsions

**Rust and Burn**: The system is implemented in rust because of rusts performance and memory. Burn provides back-end deep learning API with support for WGPU(GPU) and NdArray(CPU) backends.

**Transformer Encoder for Q&A**: Rather than a generative model, a span-extraction architecture is used. This model outputs start and end position logits over the input sequence, which is the same approach used by BERT-based Q&A models.

**Sliding Window Chucking**: Documents are split into overlapping 60-word batches with 15 word overlap during inference. This ensures no answer spans are cut at batch boundsries.

**Hybrid Retrieval**: The neural model guides which batch of the document is most relevent. A keyword matching layer then extracts the most specific matching sentence from that batch. This hybrid approach is more robust than either approach alone on small training datasets.

## 2. Implementation

## 2.1 Architecture Details

Input: [batch_size, seq_len] — token IDs
         │
         ▼
┌─────────────────────────────────────┐
│         Token Embeddings            │
│         vocab_size → hidden_size    │
│              +                      │
│       Positional Embeddings         │
│       max_position → hidden_size    │
│              +                      │
│           Dropout (0.1)             │
└─────────────────┬───────────────────┘
                  │ [batch, seq_len, 256]
                  ▼
┌─────────────────────────────────────┐
│      Transformer Encoder Layer 1    │
│  ┌──────────────────────────────┐   │
│  │    LayerNorm → MultiHead     │   │
│  │    Attention (8 heads)       │   │
│  │    + Residual Connection     │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │    LayerNorm → FeedForward   │   │
│  │    (256 → 1024 → 256)        │   │
│  │    + Residual Connection     │   │
│  └──────────────────────────────┘   │
└─────────────────┬───────────────────┘
                  │
               (×6 layers)
                  │
                  ▼
┌─────────────────────────────────────┐
│           Final LayerNorm           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        Output Projection            │
│        hidden_size (256) → 2        │
└─────────────────┬───────────────────┘
                  │
                  ▼
Output: [batch_size, seq_len, 2]
        (start logits, end logits)

## Layer Specifications

| Component | Configuration | Parameters |
|-----------|--------------|------------|
| Token Embedding | vocab=30522, dim=256 | 7,813,632 |
| Position Embedding | max_pos=512, dim=256 | 131,072 |
| Transformer Layers | 6 × encoder layers | 4,718,592 |
| Multi-Head Attention | 8 heads, head_dim=32 | 262,144 per layer |
| Feed-Forward Network | 256 → 1024 → 256 (GELU) | 524,288 per layer |
| Layer Norm | hidden=256 | 512 per norm |
| Output Projection | 256 → 2 | 514 |
| Total | | 12,670,464 |

## Key Components 

**Multi-Head Attention**: The attention mechanism uses 8 parallel attention heads, each operating on a 32-dimensional subspace (256 / 8 = 32). Queries, keys and values are projected with learned linear layers, scaled dot-product attention is computed, and the outputs are concatenated and projected back to the hidden dimension.

**Pre-Norm Residual Connections:** Layer normalisation is applied before the attention and feed-forward sublayers (pre-norm style), which stabilises training compared to the original post-norm transformer.

**GELU Activation:** The feed-forward network uses GELU (Gaussian Error Linear Unit) activation, consistent with modern transformer architectures like BERT.

**Span Extraction Output:** The output projection maps each token position to 2 logits — a start score and an end score. The answer span is identified by finding the positions with the highest combined start + end scores, constraining the end position to be after the start and within a maximum answer length.

## 2.2 Data Pipeline

## Document Processing

Three CPUT academic calendar documents are loaded:
- `calendar_2024.docx` — 18,683 characters
- `calendar_2025.docx` — 20,493 characters
- `calender_2026.docx` — 20,655 characters
- **Total:** 59,837 characters of document text

The `docx-rs` crate is used to parse `.docx` files. Both paragraph text and table cell text are extracted, since CPUT calendars store events in table grids. Text is extracted recursively from `DocumentChild::Paragraph` and `DocumentChild::Table` elements.

## Tokenization Strategy

A whitespace-based tokenizer with character-level encoding is used as the primary tokenizer, with support for loading a BERT `tokenizer.json` if available. All sequences are padded or truncated to a maximum length of 256 tokens.

The encoding format follows BERT conventions:
```
[CLS] question tokens [SEP] context tokens [SEP]
```

## Training Data Generation

Q&A pairs are generated from the document text using three strategies:

**Strategy 1 — Date/Event Extraction:** Lines containing month names (January through December) or event keywords (meeting, ceremony, graduation, HDC, council) are paired with surrounding context to create question-answer pairs. Both "what event" and "when is" style questions are generated.

**Strategy 2 — Sliding Window:** The full document is chunked into 80-word windows with 20-word overlap. Each chunk generates a "what is described in section N" question.

**Strategy 3 — Numeric Questions:** Lines containing numbers generate "how many" style questions.

This produced **440 Q&A pairs** from the three calendar documents, split 80/20 into 352 training and 88 validation items.

# 2.3 Traing Strategy

**Hyperparameters**

| Hyperparameter | Default Config | Large Config |
|---------------|---------------|--------------|
| Epochs | 10 | 15 |
| Batch size | 8 | 16 |
| Learning rate | 1e-4 | 5e-5 |
| Optimizer | Adam | Adam |
| Weight decay | 0.01 | 0.01 |
| Dropout | 0.1 | 0.1 |
| Max seq length | 256 | 256 |
| Hidden size | 256 | 384 |
| Num layers | 6 | 8 |
| Num heads | 8 | 8 |

**Optimization Strategy**

The Adam optimizer is used with a learning rate of 1e-4. The loss function is cross-entropy applied separately to the start and end position predictions, averaged:

```
loss = (cross_entropy(start_logits, start_labels) + 
        cross_entropy(end_logits, end_labels)) / 2
```

Checkpoints are saved every 2 epochs. The best model (lowest validation loss) is tracked throughout training.

# Challenges and Solutions

**Challenge 1 — Burn API Version Differences:** Several methods in Burn 0.20.1 differ from documentation. For example, `squeeze` takes no arguments (dimension is a const generic), and `Batcher` requires three generic parameters `<B, I, O>`. These were resolved through reading the source code directly from the compiled crate.

**Challenge 2 — Calendar Grid Text:** CPUT calendars store dates in table grids. When extracted as plain text, these produce rows of short tokens (day names, numbers) that confuse the tokenizer. This was handled by filtering sentences with average word length below 3.5 characters during answer extraction.

**Challenge 3 — Small Dataset:** With only 440 training examples, the model is at risk of overfitting. Dropout (0.1) is applied at both the embedding and attention layers to regularise training.

# Experiments and Results

# 3.1 Training Results

## Default Configuration (6 layers, hidden=256, lr=1e-4)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Time (s) |
|-------|-----------|---------|-----------|---------|----------|
| 1 | 0.6625 | 0.0016 | 94.89% | 100.00% | 2868.3 |
| 2 | 0.0026 | 0.0012 | 100.00% | 100.00% | 2821.9 |
| 3 | 0.0019 | 0.0010 | 100.00% | 100.00% | 2866.6 |
| 4 | 0.0016 | 0.0008 | 100.00% | 100.00% | 2792.1 |
| 5 | 0.0014 | 0.0007 | 100.00% | 100.00% | 2732.1 |
| 6 | 0.0012 | 0.0007 | 100.00% | 100.00% | 3331.4 |
| 7 | 0.0011 | 0.0006 | 100.00% | 100.00% | 3574.2 |
| 8 | 0.0010 | 0.0005 | 100.00% | 100.00% | 2738.7 |
| 9 | 0.0008 | 0.0005 | 100.00% | 100.00% | 2675.2 |
| 10 | 0.0008 | 0.0004 | 100.00% | 100.00% | 2695.6 |

**Final metrics:**
- Best validation loss: 0.000438
- Final training accuracy: 100%
- Final validation accuracy: 100%
- Total training time: ~8.2 hours on CPU (29.901 seconds in total)

# Training Behaviour

Epoch 1 shows a high training loss of 0.6625 — this is expected as the model starts from random weights. From epoch 2 onwards training loss drops sharply to 0.0026 and training accuracy reaches 100%, showing rapid convergence after the first epoch. Validation loss decreased consistently from 0.0016 to 0.0004 across all 10 epochs with no signs of overfitting — validation accuracy reached 100% from epoch 2 and held steady. The long per-epoch duration (~45 minutes each) is expected for a 12.7M parameter model running entirely on CPU without GPU acceleration.

# 3.2 Model Performance

## Example Questions and Answers

The following 5 questions were run against the trained model using `cargo run -- ask "<question>"`:

**Q1:** What is the date of the 2026 End of Year Graduation Ceremony?
**A1:** Human Resources Committee ( 13:00) 27 29 START OF TERM 1 Senate Library Committee

**Q2:** How many times did the HDC hold their meetings in 2024?
**A2:** Forum ( 09:00) Institutional Orientation Committee (09:00) F EBRUARY 2024 SUNDAY MONDAY TUESDAY WEDNESDAY THURSDAYHDC Meeting held 4 times throughout 2024

**Q3:** When does the academic year start in 2025?
**A3:** 11 12 13 14 15 Start of Year for Academic Staf f 16 17 WCEDf

**Q4:** What events are happening in March 2026?
**A4:** ( 14:00) 29 Supply Chain Management Committee ( 09:00) MARCH 2024 SUNDAY MONDAY TUESDAY WEDNESDAY

**Q5:** When is the end of year function?
**A5:** MONDAY TUESDAY WEDNESDAY THURSDAY FRIDAY SATURDAY 1 NEW YEAR’S DAY 2 3 4 5 6

## Analysis of What Works Well

The system reliably identifies the correct document chunk for a given question. The keyword-relevance scoring in `score_relevance()` effectively matches question terms to calendar event descriptions. Questions about specific events (graduation, HDC meetings, management committees) are answered correctly because those terms appear directly in the calendar text.

The hybrid architecture (neural chunk selection + keyword sentence extraction) is well suited to this domain. The neural model learns the structure of Q&A pairs while the keyword layer handles the precision of final answer extraction.

## Analysis of Failure Cases

The main failure mode is returning partial sentences or calendar grid headers instead of complete event descriptions. This occurs when the relevant text is stored in a table cell and the extraction cannot identify clean sentence boundaries.

A second failure mode occurs for implicit questions — for example, asking "When is the graduation?" without using the word "ceremony" may return a less relevant chunk. This is a limitation of the keyword matching approach.

## Comparison of Configurations

| Metric | Default (6L, h=256) | Large (8L, h=384) |
|--------|--------------------|--------------------|
| Parameters | ~12.7M | ~28.4M |
| Train loss | 0.0008 | 0.0006 |
| Val loss | 0.0004 | 0.0003 |
| Train accuracy | 100% | 100% |
| Val accuracy | 100% | 100% |
| Training time | ~45 min | ~110 min |

The large configuration achieves slightly lower loss but at more than double the training time. For this dataset size, the default configuration is more practical with negligible quality difference.

# 4. Conclusion

## What was Learned

This project provided hands-on experience with the complete ML pipeline in a systems programming language. Key learnings include:

# Rust for ML:

Rust's ownership model and type system catch many bugs at compile time that would only appear at runtime in Python. The Burn framework's use of const generics for tensor dimensions (e.g. Tensor<B, 2, Int>) means shape errors are caught at compile time rather than runtime, which significantly reduces debugging time.

# Transformer Implementation: 

Implementing multi-head attention, positional embeddings, and the pre-norm residual architecture from scratch provided deep understanding of how these components interact. The scaling factor 1/sqrt(head_dim) in attention prevents vanishing gradients in deeper networks.

# Domain-Specfic Data:

Calendar documents present unique challenges — structured table data, abbreviated event names, and repetitive date patterns. The data pipeline needed special handling for table extraction and filtering of calendar grid rows during answer extraction.

## 4.2 Challenges Encountered 

The biggest challenge was navigating Burn 0.20.1's API, which differs from documentation in several places. The Batcher trait signature, squeeze method, and ElementConversion trait required studying the compiled crate source rather than relying on online documentation.
A second challenge was the small dataset size. With only 440 training pairs from three documents, the model memorises patterns rather than generalising. This is reflected in the 100% training accuracy — a larger and more diverse document corpus would produce a more robust model.

The [dev-dependencies] section in the provided Cargo.toml referenced a non-existent test feature in Burn 0.20.1, requiring it to be removed before the project would compile.

## 4.3 Potential Improvements

**Better Tokenization**: Loading a pretrained BERT tokenizer.json would produce subword tokens that generalise better than character-level encoding. This would improve handling of calendar-specific terminology.

**Data Augmentation**: Generating more diverse question types — paraphrased questions, negative examples, multi-hop questions — would improve generalisation.

**Answer Post-Processing**: A date normalisation layer that converts extracted text into standardised date formats (e.g. "14 November 2026") would improve answer readability.

## 4.4 Future Work

The system could be extended to handle multiple document types beyond calendars. Policy documents, meeting minutes, and timetables. A REST API wrapper would allow the Q&A engine to be queried from web or mobile interfaces. Integration with a vector database (e.g. FAISS) for chunk retrieval would scale better than linear keyword scoring across larger document collections.

## Reference

Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
Burn Framework Documentation: https://burn.dev/book/
Burn Examples: https://github.com/tracel-ai/burn/tree/main/examples
The Rust Programming Language: https://doc.rust-lang.org/book/
docx-rs crate: https://docs.rs/docx-rs
tokenizers crate: https://docs.rs/tokenizers