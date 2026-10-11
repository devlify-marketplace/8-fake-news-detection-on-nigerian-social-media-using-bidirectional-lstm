import torch

from nigerian_fake_news.model import BiLSTMClassifier


def test_bilstm_forward_shapes_and_finite_values():
    model = BiLSTMClassifier(vocab_size=40, embedding_dim=12, hidden_dim=8, dropout=0.0)
    tokens = torch.tensor([[2, 3, 4], [5, 6, 0]], dtype=torch.long)
    lengths = torch.tensor([3, 2], dtype=torch.long)
    logits = model(tokens, lengths)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
