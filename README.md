# Word Doc Q&A

A Rust command-line prototype for asking questions about Cape Peninsula University of Technology (CPUT) academic calendars. It reads paragraphs and tables from three bundled Word documents covering 2024–2026. The project also implements and trains an extractive question-answering transformer using Burn.

## Try it locally

Install a current stable [Rust toolchain](https://www.rust-lang.org/tools/install) with Cargo. This project uses Rust 2024. Run the commands from the repository root so the app can find the sample documents.

```sh
git clone https://github.com/MaleselaModiba123/word-doc-qa.git
cd word-doc-qa
cargo run --locked --release -- help
cargo run --locked --release -- ask "How many times did the HDC hold their meetings in 2024?"
```

The first build downloads and compiles dependencies. The CLI prints a document preview before the `Answer:` line. The historical example in the [project report](docs/REPORT.md) found text saying that the HDC met four times in 2024, along with surrounding calendar labels. No training is required to try the demo.

## How it works

| Part | What it does |
| --- | --- |
| [Data preparation](src/data/) | Reads the bundled `.docx` files, generates question–answer pairs from the extracted text, and tokenizes them. |
| [Transformer](src/model/) | Uses a Burn encoder to predict start and end positions for answer spans. |
| [Training](src/training/) | Runs on the CPU with Burn's NdArray backend and records metrics and model checkpoints. |
| [CLI inference](src/inference/) | Returns a short excerpt using keyword matching. The CLI attempts to load a checkpoint, but the current answer path doesn't run a transformer forward pass. |

The demo reads three fixed filenames in [`docs/`](docs/). Calendar tables can lose their row and column context when flattened into text, so an excerpt may contain unrelated labels, omit the requested date, or come from the wrong year.

## Run the model or train it

Pass one of these arguments after `cargo run --locked --release --`.

| Argument | Action |
| --- | --- |
| `test` | Runs a model forward pass on dummy input as a smoke check. |
| `train-quick` | Trains the default model for 2 epochs. |
| `train` | Trains the default model for 10 epochs. |
| `train-large` | Trains the larger model for 15 epochs and writes checkpoints to `checkpoints_large/`. |

The repository includes around 145 MiB of historical checkpoints. Training writes new checkpoints and replaces the local `training_history.csv`. The recorded 10-epoch CPU run took around 8.1 hours.

## Results and limitations

The [training log](training_history.csv) records a final validation loss of 0.000438 and 100% validation accuracy for the start position on automatically generated examples. Those metrics do not measure how often the CLI answers natural-language questions correctly. The [project report](docs/REPORT.md) includes the course experiment, sample outputs, and failure cases.

Built by Malesela Modiba as a CPUT coursework project.
