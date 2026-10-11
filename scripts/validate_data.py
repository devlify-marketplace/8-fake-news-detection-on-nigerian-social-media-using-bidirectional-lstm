from __future__ import annotations

import argparse
import json
from pathlib import Path

from nigerian_fake_news.data import LABEL_NAMES, load_dataset, make_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a labelled social-post CSV without printing post text")
    parser.add_argument("csv", nargs="?", default="data/raw/posts.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = load_dataset(args.csv)
    splits = make_splits(frame, seed=args.seed)
    report = {
        "validated_rows_after_exact_deduplication": int(len(frame)),
        "independent_groups": int(frame["_group"].nunique()),
        "class_counts": {LABEL_NAMES[int(k)]: int(v) for k, v in frame["label"].value_counts().sort_index().items()},
        "split_sizes": {name: int(len(getattr(splits, name))) for name in ("train", "validation", "test")},
        "split_class_counts": {
            name: {LABEL_NAMES[int(k)]: int(v) for k, v in getattr(splits, name)["label"].value_counts().sort_index().items()}
            for name in ("train", "validation", "test")
        },
        "columns": [c for c in frame.columns if not c.startswith("_") and c != "normalized_text"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
