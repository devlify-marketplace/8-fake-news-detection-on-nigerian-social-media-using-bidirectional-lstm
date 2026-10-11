import pandas as pd
import pytest

from nigerian_fake_news.data import load_dataset, make_splits


def test_load_and_split_groups_without_leakage(tmp_path):
    rows = []
    for label in (0, 1):
        for i in range(15):
            rows.append({"text": f"post text class {label} event {i}", "label": label, "event_id": f"{label}-{i}"})
    rows.append(rows[0].copy())  # exact duplicate must be removed
    path = tmp_path / "posts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame = load_dataset(path)
    assert len(frame) == 30
    split = make_splits(frame, seed=7)
    group_sets = [set(part["_group"]) for part in (split.train, split.validation, split.test)]
    assert not group_sets[0] & group_sets[1]
    assert not group_sets[0] & group_sets[2]
    assert not group_sets[1] & group_sets[2]
    assert all(set(part["label"]) == {0, 1} for part in (split.train, split.validation, split.test))


def test_rejects_verification_badge_as_truth_label(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"text": "one", "label": "verified"}, {"text": "two", "label": "unverified"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unsupported label"):
        load_dataset(path)


def test_rejects_conflicting_labels_for_same_text(tmp_path):
    path = tmp_path / "conflict.csv"
    pd.DataFrame([
        {"text": "Same claim text", "label": "credible"},
        {"text": " same   claim TEXT ", "label": "misleading"},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting truth labels"):
        load_dataset(path)


def test_requires_both_classes(tmp_path):
    path = tmp_path / "one_class.csv"
    pd.DataFrame([{"text": f"post {i}", "label": 0} for i in range(8)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Both classes"):
        load_dataset(path)
