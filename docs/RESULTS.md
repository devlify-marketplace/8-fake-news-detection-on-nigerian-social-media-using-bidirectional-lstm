# Results and Interpretation

## Status: empirical evaluation pending

No training run was performed and no metric is reported. The available partial corpus considered during project implementation has missing or invalid truth labels and is excluded; see [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md). The project's test fixtures are synthetic software fixtures and are never used for model training or research results.

After an approved labelled dataset is supplied, run:

```bash
python scripts/validate_data.py data/raw/posts.csv
python -m nigerian_fake_news.train --root . --data data/raw/posts.csv
```

The training process writes `metrics_bilstm.json`, `metrics_tfidf_baseline.json`, `split_summary.csv`, `training_history.csv`, `test_predictions.csv`, and diagnostic plots beneath the timestamped run directory in `artifacts/`.

## Reporting checklist

Before completing this section, report:

- Source, collection dates, permission/license, sample definition, language/platform coverage, and annotation process.
- Raw and retained counts, duplicate rate, class distribution, excluded/ambiguous data, inter-annotator agreement, and claim/event groups.
- Split unit and exact train/validation/test counts; demonstrate no claim/event or near-duplicate leakage.
- BiLSTM and TF-IDF baseline results on the identical untouched grouped test set: accuracy, balanced accuracy, misleading-class precision/recall/F1, macro F1, MCC, ROC-AUC, confusion matrix, plus confidence intervals when supported.
- Error analysis by topic/platform/language and a discussion of source/topic shortcuts and temporal generalization.
- A clear statement that text classification is not evidence-based fact verification.

**Do not fill this section with illustrative/synthetic metrics.**
