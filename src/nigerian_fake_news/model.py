from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    """Word-embedding + one-layer bidirectional LSTM + binary-logit head."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 64,
        dropout: float = 0.35,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must include <pad> and <unk>")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 1))

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or lengths.ndim != 1:
            raise ValueError("Expected token_ids [batch, seq] and lengths [batch]")
        if token_ids.shape[0] != lengths.shape[0]:
            raise ValueError("Batch size of token_ids and lengths must match")
        embedded = self.input_dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu().clamp(min=1, max=token_ids.shape[1]),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        # Shape: [layers * directions, batch, hidden]
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.head(features).squeeze(1)
