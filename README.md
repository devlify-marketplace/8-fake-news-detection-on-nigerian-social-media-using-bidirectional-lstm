# Fake News Detection on Nigerian Social Media Using Bidirectional LSTM

**Topic 8** from the supplied final-year project list. This is a runnable, research-oriented BiLSTM text-classification system with a local API and web demo. It is built to distinguish **credible (0)** from **misleading (1)** posts, but it is **not trained or presented with fake results** because the available Nigerian social-media corpus examined for this project does not provide sound claim-veracity labels. Supply an approved labelled dataset to train it. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) before any data collection.

> **Research-only warning.** A text classifier is not a fact-checker. Its score is uncalibrated, can reflect topical or source artifacts, and must not be used to label a person or make high-impact decisions. Verify a claim with evidence and human review.

## Included

- `src/nigerian_fake_news/model.py` — word embedding + bidirectional LSTM + binary classification head.
- `src/nigerian_fake_news/data.py` — CSV validation, strict truth-label mapping, duplicate removal, event/claim-aware 60/20/20 split, leakage checks.
- `src/nigerian_fake_news/train.py` — deterministic training, class-weighted loss, validation early stopping, held-out test evaluation, and same-split TF-IDF/logistic-regression baseline.
- `src/nigerian_fake_news/api.py` and `web/index.html` — local inference API and paste-a-post demo. The API refuses prediction until a trained model exists and does not save submitted text.
- `scripts/validate_data.py` — preflight data check without printing post contents.
- `tests/` — offline engineering tests. Test fixtures are synthetic and are used **only** for software tests, never model fitting or reported research results.
- `docs/` — academic project report, ethics/data collection protocol, provenance, model card, results status, and architecture diagram.

## Install

Python 3.10–3.12; CPU works, CUDA is optional.

```bash
cd /workspace/topic-08-fake-news-detection-bilstm
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[test]'
```

## Prepare data

Create `data/raw/posts.csv` with at least these columns:

```csv
text,label,platform,source_url,claim_id
"Example post text",credible,x,https://source.example/post/1,claim-001
"Another example post",misleading,facebook,https://source.example/post/2,claim-002
```

Labels must mean factual credibility, not whether an account is verified: `0` / `credible` / `real` / `true` maps to class 0; `1` / `misleading` / `fake` / `false` maps to class 1. Both classes need at least five independent claim/event groups. `claim_id`, `event_id`, or `group_id` is strongly recommended so related posts do not leak between train, validation and test partitions. The exact input schema and collection safeguards are in [`data/raw/README.md`](data/raw/README.md) and [`docs/DATA_COLLECTION_PROTOCOL.md`](docs/DATA_COLLECTION_PROTOCOL.md).

No raw dataset is supplied. Do not use a platform verification badge as a truth label, and do not use the publicly linked corpus described in the provenance notes as a fake/credible benchmark.

## Validate, train, and evaluate

```bash
python scripts/validate_data.py data/raw/posts.csv
python -m nigerian_fake_news.train --root . --data data/raw/posts.csv
```

The training script creates a group-separated 60/20/20 split (seed 42), constructs vocabulary from training text only, saves the best checkpoint by validation loss, and evaluates once on the untouched test partition. It reports accuracy, balanced accuracy, precision/recall/F1 for the misleading class, macro F1, MCC, ROC-AUC, confusion matrix, plus a TF-IDF/logistic-regression baseline using the **same split**. Generated results go under `artifacts/run_*`; `artifacts/model/` contains the latest model/vocabulary for inference. Raw posts are not written into the predictions file. Review the dataset's license, collection permission, annotation quality, split summary, and error cases before interpreting any metrics.

Optional quick engineering run after data is added: `python -m nigerian_fake_news.train --root . --data data/raw/posts.csv --epochs 1`. Such a smoke test is not reportable research evidence.

## Run the demo

```bash
uvicorn nigerian_fake_news.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Routes: `GET /api/health`, `POST /api/predict` with JSON `{"text":"..."}`. Without a trained checkpoint, health reports `model_ready: false` and prediction returns HTTP 503 rather than an invented result. When trained, API responses provide an **uncalibrated misleading score** and an abstain/review option. The service does not persist submitted text or prediction history.

## Tests

```bash
pytest -q
```

Tests do not download social-media data or train the model. They verify preprocessing, schema/label handling, group leakage checks, model output shape, and the untrained API behavior.

## Research files

- [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)
- [`docs/DATA_COLLECTION_PROTOCOL.md`](docs/DATA_COLLECTION_PROTOCOL.md)
- [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- [`docs/RESULTS.md`](docs/RESULTS.md)
- [`docs/ARCHITECTURE.mmd`](docs/ARCHITECTURE.mmd)

The delivered implementation is complete and testable; trained model weights, empirical metrics, and a data-specific findings section remain intentionally pending an approved, properly labelled corpus.
