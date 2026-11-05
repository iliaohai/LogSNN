# LogSNN

Code accompanying **[LogSNN: Spiking Neural Networks for System Log Anomaly Detection](https://ssrn.com/abstract=6343052)** by Song Chen, Hai Liao, Yan Liang, Hang Zhou, and Fan Min.

LogSNN classifies log-event sequences using a pretrained text encoder, a two-class linear layer, and a leaky integrate-and-fire (LIF) neuron. This repository contains **code only**; datasets, pretrained models, trained checkpoints, and experiment logs are not distributed.

## 1. Install

Use Linux and Python 3.8. The dependencies below match the original server environment. An NVIDIA GPU is recommended for full training; CPU execution is supported by `train.py`.

```bash
git clone https://github.com/iliaohai/LogSNN.git
cd LogSNN
conda create -n logsnn python=3.8 -y
conda activate logsnn

# Choose ONE PyTorch build:
python -m pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# CPU alternative:
# python -m pip install torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

See [PyTorch's installation archive](https://pytorch.org/get-started/previous-versions/) for platform requirements. Other dependency versions have not been validated.

## 2. Prepare inputs

Obtain HDFS/BGL logs from [Loghub](https://github.com/logpai/loghub) and prepare **sequence-level CSVs** locally, or use your existing prepared CSVs. Required columns:

| Dataset | Suggested local path | Sequence column | Label column |
| --- | --- | --- | --- |
| HDFS | `dataset/HDFS/data_instances2.csv` | `EventSequence` | `Label` |
| BGL | `dataset/BGL/BGL_sequence.csv` | `sequence` | `label` |

Each sequence cell is a quoted Python-list-style string of event IDs, such as `"[1, 2, 3]"`; labels must be numeric **0 = normal, 1 = anomaly**. Additional columns are ignored. Preserve event order and use one consistent event-ID mapping. Raw `.log` files cannot be passed directly to `train.py`.

The recovered source does not include raw-log parsing, event-ID mapping, or session/window construction. Consequently, the exact paper CSVs cannot be reconstructed from raw logs with this repository alone; independently prepared sequences may produce different results. No dataset is downloaded automatically.

The default model is [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small), downloaded by Transformers on first use. To work offline, pass `--model /path/to/t5-small`; the directory must contain the model configuration, PyTorch weights, and tokenizer files (`spiece.model` for T5). Keep model files outside Git.

## 3. Check the pipeline

Run one epoch on the first 32 rows of a prepared CSV:

```bash
python train.py --dataset hdfs --data dataset/HDFS/data_instances2.csv \
  --device cpu --epochs 1 --batch-size 2 --limit 32 \
  --output runs/hdfs-smoke
```

Use a new output directory for each run. This checks loading, forward/backward passes, evaluation, and checkpoint writing; the subset is not a benchmark and may contain only one class.

## 4. Train and evaluate

```bash
# Defaults recovered from the server's T5-small scripts:
python train.py --dataset hdfs --data dataset/HDFS/data_instances2.csv \
  --device cuda --epochs 32 --batch-size 620 --output runs/hdfs

python train.py --dataset bgl --data dataset/BGL/BGL_sequence.csv \
  --device cuda --epochs 320 --batch-size 160 --output runs/bgl
```

Reduce `--batch-size` if GPU memory is insufficient; this changes the optimization trajectory. Use `--model google-t5/t5-base` to select T5-base, or `--backbone bert --model bert-base-uncased` for BERT. The classifier width follows the encoder configuration.

The default protocol uses an 80/20 random sequence split (`--test-size 0.2`, `--seed 32`), learning rate `2e-5`, Transformers AdamW, cross-entropy loss, and gradient clipping at 1.0. Training order is fixed. The same held-out split is evaluated after each epoch and selects the best checkpoint by `(accuracy + F1) / 2`; it is a **validation split**, not an independent final test set. F1 uses anomaly class 1; precision and recall are macro-averaged. For independent reporting, reserve a separate test set before model selection.

Each run saves:

- `config.json`: arguments, input-file SHA-256, row counts, and PyTorch version.
- `split_indices.npz`: zero-based training and validation row indices.
- `metrics.jsonl`: loss and validation metrics per epoch.
- `best.pt`, `best_metrics.json`, `best_predictions.csv`: selected model and its validation results.
- `tokenizer/`, `encoder_config/`: tokenizer and encoder configuration.

Seeds and deterministic PyTorch operations are enabled; exact numerical agreement across hardware/library versions is not guaranteed. Keep the CSV, model snapshot, environment, and command fixed when comparing runs. `python train.py --help` lists all options.

## Implementation scope

`model.py` and `train.py` provide a configurable version of the recovered server implementation. They retain first-token pooling, LIF settings (`beta=1`, threshold 1, fast-sigmoid slope 25), hidden-state reset on every forward pass, membrane-potential logits, and the original text transformation: slice `[1:len(sequence)-2]`, split on commas, remove spaces/single quotes, cap at 510 characters, and wrap with `[CLS]` / `[SEB]`. The unusual slice and `[SEB]` spelling are preserved for compatibility. The implementation performs **one LIF step per forward pass**; it does not implement a separate multi-step spike encoder or energy measurement.

The maintained entry point adds CPU support, tokenizer truncation at 512 tokens, automatic classifier width, seeded initialization, saved outputs, and processing of the final partial training batch. The server scripts dropped that partial batch and seeded only the split. These changes can affect scores.

The paper describes T5-base; the recovered server scripts default to T5-small. This release documents the available implementation and does not claim exact reproduction of the paper's reported scores or energy results.

Validation: one-epoch CPU smoke runs on 32 prepared rows from each dataset, using the server's local T5-small weights and dependency versions above. Full GPU training and a fresh environment installation have not been validated.

| Path | Contents |
| --- | --- |
| `model.py`, `train.py` | Maintained training and evaluation entry point |
| `legacy/server/` | Five original server source files, preserved unchanged |
| `legacy/github/` | Previous GitHub BERT-only scripts, preserved unchanged |

Legacy scripts use hard-coded CUDA devices, relative paths, and output directories; use the maintained entry point for the tutorial above.

## Citation

```bibtex
@misc{chen2026logsnn,
  title = {{LogSNN}: Spiking Neural Networks for System Log Anomaly Detection},
  author = {Chen, Song and Liao, Hai and Liang, Yan and Zhou, Hang and Min, Fan},
  year = {2026},
  publisher = {SSRN},
  doi = {10.2139/ssrn.6343052},
  url = {https://ssrn.com/abstract=6343052},
  note = {Available at SSRN 6343052}
}
```

Also available as [`citation.bib`](citation.bib).

## License

Source code is licensed under [Apache-2.0](LICENSE). Datasets and pretrained models remain subject to their respective licenses.
