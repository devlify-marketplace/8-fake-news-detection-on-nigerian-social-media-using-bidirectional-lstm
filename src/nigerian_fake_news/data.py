from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .preprocessing import normalized_key

LABEL_MAP = {
    "0": 0, "0.0": 0, "credible": 0, "real": 0, "true": 0, "mostly true": 0,
    "1": 1, "1.0": 1, "misleading": 1, "fake": 1, "false": 1, "mostly false": 1,
}
LABEL_NAMES = {0: "credible", 1: "misleading"}
REQUIRED_COLUMNS = {"text", "label"}


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _map_label(value: object) -> int:
    key = str(value).strip().casefold()
    if key not in LABEL_MAP:
        raise ValueError(
            f"Unsupported label {value!r}. Use 0/credible/real/true or "
            "1/misleading/fake/false; do not use account verification as a truth label."
        )
    return LABEL_MAP[key]


def load_dataset(path: str | Path, deduplicate: bool = True) -> pd.DataFrame:
    """Load labelled posts; optionally remove exact normalized duplicates."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame = frame[frame["text"].str.len() > 0].copy()
    if frame.empty:
        raise ValueError("No non-empty text rows found")
    try:
        frame["label"] = frame["label"].map(_map_label).astype(int)
    except ValueError as exc:
        raise ValueError(f"Invalid label in dataset: {exc}") from exc
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Both classes are required: label 0=credible, label 1=misleading")
    frame["normalized_text"] = frame["text"].map(normalized_key)
    conflicting = frame.groupby("normalized_text")["label"].nunique()
    if (conflicting > 1).any():
        raise ValueError("The same normalized post has conflicting truth labels; adjudicate duplicates before training")
    if deduplicate:
        frame = frame.drop_duplicates(subset=["normalized_text"], keep="first").copy()
    # Prefer human-assigned event/claim grouping to exact-text grouping when present.
    group_column = next((c for c in ("group_id", "claim_id", "event_id") if c in frame.columns), None)
    if group_column:
        supplied = frame[group_column].astype(str).str.strip()
        frame["_group"] = [
            value if value and value.casefold() not in {"nan", "none", "null"}
            else hashlib.sha256(key.encode("utf-8")).hexdigest()
            for value, key in zip(supplied, frame["normalized_text"])
        ]
    else:
        frame["_group"] = frame["normalized_text"].map(
            lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        )
    frame = frame.reset_index(drop=True)
    return frame


def make_splits(frame: pd.DataFrame, seed: int = 42) -> DatasetSplits:
    """Create a deterministic 60/20/20 grouped-stratified split.

    A supplied `group_id`, `claim_id`, or `event_id` keeps related variants in a
    single partition. Exact normalized text is the fallback group. At least five
    independent groups per class are required for a stable five-fold split.
    """
    if "_group" not in frame or "label" not in frame:
        raise ValueError("Expected a validated frame from load_dataset()")
    groups = frame["_group"].astype(str).to_numpy()
    y = frame["label"].astype(int).to_numpy()
    group_counts = frame.groupby("label")["_group"].nunique()
    if len(group_counts) != 2 or int(group_counts.min()) < 5:
        raise ValueError(
            "Need at least 5 independent claim/event groups for each class; "
            f"found {group_counts.to_dict()}"
        )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(frame, y, groups))
    train_val_idx, test_idx = folds[0]
    train_val = frame.iloc[train_val_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    tv_groups = train_val["_group"].astype(str).to_numpy()
    tv_y = train_val["label"].astype(int).to_numpy()
    val_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_idx, validation_idx = next(val_splitter.split(train_val, tv_y, tv_groups))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    validation = train_val.iloc[validation_idx].reset_index(drop=True)
    if any(part["label"].nunique() != 2 for part in (train, validation, test)):
        raise ValueError("A split contains only one class; add independent examples per class")
    sets = [set(part["_group"]) for part in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("Group leakage detected between dataset partitions")
    return DatasetSplits(train, validation, test)
