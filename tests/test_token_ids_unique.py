import pytest

import tiktoken


@pytest.mark.parametrize("enc_name", tiktoken.list_encoding_names())
def test_special_token_ids_unique(enc_name):
    enc = tiktoken.get_encoding(enc_name)
    seen = {}
    for name, token_id in enc._special_tokens.items():
        assert token_id not in seen, (
            f"{enc_name}: special tokens {seen[token_id]!r} and {name!r} "
            f"share token id {token_id}"
        )
        seen[token_id] = name


@pytest.mark.parametrize("enc_name", tiktoken.list_encoding_names())
def test_mergeable_token_ids_unique(enc_name):
    enc = tiktoken.get_encoding(enc_name)
    token_ids = list(enc._mergeable_ranks.values())
    assert len(token_ids) == len(set(token_ids)), f"{enc_name}: duplicate mergeable token ids"
