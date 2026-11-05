# LogSNN

Code for [LogSNN: Spiking Neural Networks for System Log Anomaly Detection](https://ssrn.com/abstract=6343052).

## 1. Install dependencies

Use Linux, Python 3.8, and an NVIDIA GPU.

```bash
git clone https://github.com/iliaohai/LogSNN.git
cd LogSNN
conda create -n logsnn python=3.8 -y
conda activate logsnn
python -m pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements.txt
```

## 2. Prepare the data

Place your prepared CSV files in these folders:

```text
dataset/
├── HDFS/data_instances2.csv
└── BGL/BGL_sequence.csv
```

- HDFS requires columns `EventSequence` and `Label`.
- BGL requires columns `sequence` and `label`.
- Sequences are list-style event IDs, such as `"[1, 2, 3]"`. Labels are `0` (normal) or `1` (anomaly).

Raw logs are available from [Loghub](https://github.com/logpai/loghub). Convert them to sequence CSVs before training; raw-log preprocessing is not included. Datasets and model weights are not included in this repository.

## 3. Run training

The default T5-small model is downloaded automatically on the first run.

```bash
# HDFS
python train.py --dataset hdfs --data dataset/HDFS/data_instances2.csv \
  --device cuda --epochs 32 --batch-size 620 --output runs/hdfs

# BGL
python train.py --dataset bgl --data dataset/BGL/BGL_sequence.csv \
  --device cuda --epochs 320 --batch-size 160 --output runs/bgl
```

Reduce `--batch-size` if GPU memory is insufficient. For a local model, add `--model pre-models/t5-small`. For T5-base, add `--model google-t5/t5-base`.

## 4. View results

Training evaluates an 80/20 train/validation split after each epoch. Results are saved in the specified output folder:

- `metrics.jsonl`: metrics for each epoch.
- `best_metrics.json`: best validation results.
- `best.pt`: best model checkpoint.
- `best_predictions.csv`: predictions from the best epoch.

Use a new output folder for each run. Run `python train.py --help` for more options.

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

## License

[Apache-2.0](LICENSE).
