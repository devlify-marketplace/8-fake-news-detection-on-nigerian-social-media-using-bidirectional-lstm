from nigerian_fake_news.preprocessing import Vocabulary, normalized_key, tokenize


def test_normalization_replaces_links_and_handles_mentions():
    tokens = tokenize("Naija News @reporter https://example.ng/post #EndSARS")
    assert "usertoken" in tokens
    assert "urltoken" in tokens
    assert "#endsars" in tokens
    assert normalized_key("Hello   World") == normalized_key("HELLO world")


def test_vocabulary_is_deterministic_and_has_reserved_ids():
    a = Vocabulary.build(["Nigerian news", "news is here"], max_size=10)
    b = Vocabulary.build(["Nigerian news", "news is here"], max_size=10)
    assert a.token_to_id == b.token_to_id
    assert a.token_to_id["<pad>"] == 0
    assert a.token_to_id["<unk>"] == 1
    assert a.encode("outside vocabulary", max_length=10)[0] == 1
