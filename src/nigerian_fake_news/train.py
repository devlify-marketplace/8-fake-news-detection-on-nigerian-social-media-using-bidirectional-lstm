from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import LABEL_NAMES, load_dataset, make_splits
from .metrics import binary_metrics
from .model import BiLSTMClassifier
from .preprocessing import Vocabulary


@dataclass
class TrainConfig:
    seed: int = 42
    max_vocab_size: int = 30000
    min_token_frequency: int = 1
    max_sequence_length: int = 160
    embedding_dim: int = 100
    hidden_dim: int = 64
    dropout: float = 0.35
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 4
    decision_threshold: float = 0.5


class PostDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: Vocabulary, max_length: int):
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        ids = self.vocab.encode(self.texts[index], self.max_length)
        return ids, self.labels[index]


def _collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    width = max(lengths).item()
    token_ids = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        token_ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return token_ids, lengths, torch.tensor(labels, dtype=torch.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _predict(model, loader, device):
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for token_ids, lengths, target in loader:
            logits = model(token_ids.to(device), lengths)
            scores.extend(torch.sigmoid(logits).cpu().numpy().astype(float).tolist())
            labels.extend(target.numpy().astype(int).tolist())
    return labels, scores


def _plot_history(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Weighted BCE loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["validation_macro_f1"] for r in history], color="#16817a")
    axes[1].set(title="Validation macro F1", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(metrics: dict, path: Path) -> None:
    cm = np.asarray(metrics["confusion_matrix_labels_0_credible_1_misleading"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Credible", "Misleading"])
    ax.set_yticks([0, 1], labels=["Credible", "Misleading"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Reference label")
    ax.set_title("Held-out test confusion matrix")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_project(data_path: str | Path, root: str | Path, config: TrainConfig) -> Path:
    root = Path(root).resolve()
    data_path = Path(data_path).resolve()
    _set_seed(config.seed)
    frame = load_dataset(data_path)
    splits = make_splits(frame, seed=config.seed)
    vocab = Vocabulary.build(
        splits.train["text"],
        max_size=config.max_vocab_size,
        min_frequency=config.min_token_frequency,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        "train": DataLoader(PostDataset(splits.train, vocab, config.max_sequence_length), batch_size=config.batch_size, shuffle=True, collate_fn=_collate),
        "validation": DataLoader(PostDataset(splits.validation, vocab, config.max_sequence_length), batch_size=config.batch_size, shuffle=False, collate_fn=_collate),
        "test": DataLoader(PostDataset(splits.test, vocab, config.max_sequence_length), batch_size=config.batch_size, shuffle=False, collate_fn=_collate),
    }
    model = BiLSTMClassifier(
        len(vocab.token_to_id), config.embedding_dim, config.hidden_dim, config.dropout
    ).to(device)
    negatives = int((splits.train["label"] == 0).sum())
    positives = int((splits.train["label"] == 1).sum())
    if not positives or not negatives:
        raise ValueError("Training partition must contain both labels")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = root / "artifacts" / run_name
    model_dir = root / "artifacts" / "model"
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "weights.pt"
    best_val = float("inf")
    wait = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for token_ids, lengths, target in loaders["train"]:
            token_ids, target = token_ids.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids, lengths)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * target.size(0)
            seen += target.size(0)
        train_loss = total_loss / max(1, seen)
        val_y, val_scores = _predict(model, loaders["validation"], device)
        val_pred = np.asarray(val_scores) >= config.decision_threshold
        val_loss_total, val_seen = 0.0, 0
        model.eval()
        with torch.no_grad():
            for ids, lengths, target in loaders["validation"]:
                logits = model(ids.to(device), lengths)
                loss = criterion(logits, target.to(device))
                val_loss_total += loss.item() * target.size(0)
                val_seen += target.size(0)
        val_loss = val_loss_total / max(1, val_seen)
        from sklearn.metrics import f1_score
        val_f1 = float(f1_score(val_y, val_pred.astype(int), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss, "validation_macro_f1": val_f1})
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}", flush=True)
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= config.patience:
                break

    # Evaluate exactly once on the untouched, group-separated test partition.
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    y_test, scores_test = _predict(model, loaders["test"], device)
    neural_metrics = binary_metrics(y_test, scores_test, config.decision_threshold)
    pd.DataFrame({
        "row_index_in_validated_dataset": splits.test.index,
        "true_label": y_test,
        "misleading_score_uncalibrated": scores_test,
        "predicted_label": (np.asarray(scores_test) >= config.decision_threshold).astype(int),
    }).to_csv(run_dir / "test_predictions.csv", index=False)

    # TF-IDF + logistic regression is the classical baseline on the same partitions.
    baseline_vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_train = baseline_vec.fit_transform(splits.train["text"])
    baseline = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=config.seed)
    baseline.fit(x_train, splits.train["label"].astype(int))
    baseline_scores = baseline.predict_proba(baseline_vec.transform(splits.test["text"]))[:, 1]
    baseline_metrics = binary_metrics(splits.test["label"].astype(int).tolist(), baseline_scores, config.decision_threshold)
    import joblib
    joblib.dump({"vectorizer": baseline_vec, "model": baseline}, run_dir / "tfidf_logistic_baseline.joblib")

    metadata = {
        "architecture": "Embedding(100) -> BiLSTM(hidden=64, bidirectional=True) -> dropout -> sigmoid logit",
        "label_map": {"0": "credible", "1": "misleading"},
        "vocab_size": len(vocab.token_to_id),
        "max_sequence_length": config.max_sequence_length,
        "decision_threshold": config.decision_threshold,
        "abstain_threshold": 0.65,
        "config": asdict(config),
        "training_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "training_rows_after_validation_and_deduplication": int(len(frame)),
        "split_sizes": {name: int(len(getattr(splits, name))) for name in ("train", "validation", "test")},
        "label_names": LABEL_NAMES,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_are_from": "single group-separated held-out test partition; not externally validated",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "vocabulary.json").write_text(json.dumps(vocab.to_json(), ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics_bilstm.json").write_text(json.dumps(neural_metrics, indent=2), encoding="utf-8")
    (run_dir / "metrics_tfidf_baseline.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    split_rows = []
    for name in ("train", "validation", "test"):
        part = getattr(splits, name)
        for label, count in part["label"].value_counts().sort_index().items():
            split_rows.append({"split": name, "label": LABEL_NAMES[int(label)], "count": int(count)})
    pd.DataFrame(split_rows).to_csv(run_dir / "split_summary.csv", index=False)
    _plot_history(history, run_dir / "training_history.png")
    _plot_confusion(neural_metrics, run_dir / "confusion_matrix_bilstm.png")
    (root / "artifacts" / "LATEST_RUN.txt").write_text(run_name + "\n", encoding="utf-8")
    print(f"Saved model to {model_dir}; results to {run_dir}")
    print(json.dumps({"bilstm_test": neural_metrics, "tfidf_baseline_test": baseline_metrics}, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Topic 8 BiLSTM project")
    parser.add_argument("--data", default="data/raw/posts.csv", help="CSV containing text and verified binary label")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    for key in ("epochs", "batch_size", "seed"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    train_project(args.data if Path(args.data).is_absolute() else root / args.data, root, TrainConfig(**cfg))


if __name__ == "__main__":
    main()
