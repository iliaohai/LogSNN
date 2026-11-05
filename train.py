"""Train and evaluate LogSNN on a prepared HDFS or BGL sequence CSV."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AdamW

from model import LogSNN


def format_sequence(value):
    # Preserve the server scripts' slicing, character cap, and marker spelling.
    value = str(value)
    events = value[1:len(value) - 2].split(",")
    text = "".join(event.replace(" ", "").replace("'", "") + " " for event in events)
    return "[CLS]" + text[:510] + "[SEB]"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hdfs", "bgl"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="google-t5/t5-small", help="HF ID or local model directory")
    parser.add_argument("--backbone", choices=["t5", "bert"], default="t5")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--limit", type=int, help="Read only the first N rows for a smoke test")
    parser.add_argument("--output", type=Path, required=True, help="New, empty run directory")
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (32 if args.dataset == "hdfs" else 320)
    args.batch_size = args.batch_size if args.batch_size is not None else (620 if args.dataset == "hdfs" else 160)
    if min(args.epochs, args.batch_size, args.eval_batch_size) < 1:
        parser.error("epochs and batch sizes must be positive")
    if not 0 < args.test_size < 1 or (args.limit is not None and args.limit < 4):
        parser.error("test-size must be in (0, 1); limit must be at least 4")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is not empty; choose a new directory")
    return args


def main():
    args = arguments()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or install a compatible CUDA build")

    frame = pd.read_csv(args.data, nrows=args.limit)
    sequence, label = ("EventSequence", "Label") if args.dataset == "hdfs" else ("sequence", "label")
    if not {sequence, label}.issubset(frame.columns):
        raise ValueError("Required CSV columns: %s, %s" % (sequence, label))
    if frame[[sequence, label]].isna().any().any() or not frame[label].isin([0, 1]).all():
        raise ValueError("Sequences/labels must be non-null and labels must be numeric 0 or 1")
    if len(frame) < 4:
        raise ValueError("At least four rows are required")
    train_idx, eval_idx = train_test_split(
        np.arange(len(frame)), test_size=args.test_size, random_state=args.seed,
    )
    texts = [format_sequence(value) for value in frame[sequence]]
    labels = frame[label].to_numpy(dtype=np.int64)
    model = LogSNN(args.model, args.backbone).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=device, data_sha256=file_sha256(args.data), rows=len(frame),
                  train_rows=len(train_idx), eval_rows=len(eval_idx), torch_version=torch.__version__)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez(args.output / "split_indices.npz", train=train_idx, evaluation=eval_idx)
    model.tokenizer.save_pretrained(args.output / "tokenizer")
    model.encoder.config.save_pretrained(args.output / "encoder_config")
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for start in range(0, len(train_idx), args.batch_size):
            indices = train_idx[start:start + args.batch_size]
            target = torch.tensor(labels[indices], dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits = model([texts[index] for index in indices])
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(eval_idx), args.eval_batch_size):
                indices = eval_idx[start:start + args.eval_batch_size]
                predictions.extend(model([texts[index] for index in indices]).argmax(1).cpu().tolist())
        truth = labels[eval_idx]
        metrics = dict(
            epoch=epoch, train_loss=loss_sum / len(train_idx),
            accuracy=accuracy_score(truth, predictions),
            f1=f1_score(truth, predictions, zero_division=0),
            precision_macro=precision_score(truth, predictions, average="macro", zero_division=0),
            recall_macro=recall_score(truth, predictions, average="macro", zero_division=0),
        )
        print(json.dumps(metrics), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        score = (metrics["accuracy"] + metrics["f1"]) / 2
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "config": config, "metrics": metrics}, args.output / "best.pt")
            pd.DataFrame({"row_index": eval_idx, "label": truth, "prediction": predictions}).to_csv(
                args.output / "best_predictions.csv", index=False,
            )
            (args.output / "best_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
