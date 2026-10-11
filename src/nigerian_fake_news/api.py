from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .model import BiLSTMClassifier
from .preprocessing import Vocabulary

ROOT = Path(os.environ.get("NFD_ROOT", Path(__file__).resolve().parents[2])).resolve()
MODEL_DIR = ROOT / "artifacts" / "model"
WEB_FILE = ROOT / "web" / "index.html"

app = FastAPI(title="Nigerian Social Media Fake News Detection", version="1.0.0")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=5, max_length=5000)


@lru_cache(maxsize=1)
def _load_runtime():
    metadata_path = MODEL_DIR / "metadata.json"
    vocab_path = MODEL_DIR / "vocabulary.json"
    weights_path = MODEL_DIR / "weights.pt"
    if not (metadata_path.is_file() and vocab_path.is_file() and weights_path.is_file()):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vocab = Vocabulary.from_json(json.loads(vocab_path.read_text(encoding="utf-8")))
    config = metadata["config"]
    model = BiLSTMClassifier(
        vocab_size=metadata["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, vocab, metadata


@app.get("/")
def index():
    if not WEB_FILE.is_file():
        raise HTTPException(status_code=404, detail="Web interface not found")
    return FileResponse(WEB_FILE)


@app.get("/api/health")
def health():
    ready = _load_runtime() is not None
    return {
        "status": "ready" if ready else "awaiting_approved_dataset_and_training",
        "model_ready": ready,
        "message": None if ready else "No trained model is bundled. Add an approved labelled corpus and run the training command.",
    }


@app.post("/api/predict")
def predict(payload: PredictionRequest):
    runtime = _load_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="No trained model is available. Supply an approved labelled dataset and train the model first.")
    model, vocab, metadata = runtime
    ids = vocab.encode(payload.text, metadata["max_sequence_length"])
    token_ids = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    with torch.inference_mode():
        score = float(torch.sigmoid(model(token_ids, lengths)).item())
    threshold = float(metadata.get("decision_threshold", 0.5))
    abstain = float(metadata.get("abstain_threshold", 0.65))
    if max(score, 1.0 - score) < abstain:
        decision = "needs human review"
    else:
        decision = "likely misleading" if score >= threshold else "likely credible"
    return {
        "decision": decision,
        "misleading_score_uncalibrated": round(score, 4),
        "threshold": threshold,
        "disclaimer": "This text classifier is not a fact-checker. Scores are uncalibrated and can be wrong; verify claims with reliable sources and human review.",
    }
