from __future__ import annotations

import copy
import json

import pytest

import tiktoken
from tiktoken.load import (
    _bytes_to_unicode,
    hf_tokenizer_to_encoding_args,
    load_hf_tokenizer,
)

# The tests must run without network access, so instead of downloading a real tokenizer.json we
# build a small byte level BPE tokenizer in the same format. It has one token per byte plus a few
# merges, which is enough to exercise the conversion.
_BYTE_TO_UNICODE = _bytes_to_unicode()


def _token_str(token: bytes) -> str:
    return "".join(_BYTE_TO_UNICODE[b] for b in token)


def make_tokenizer_json(merges: list[str] | None = None) -> dict:
    vocab = {_token_str(bytes([b])): b for b in range(2**8)}
    if merges is None:
        merges = []
        for i, token in enumerate([b"ab", b"abc", b"hello", b" hello", b" world"]):
            vocab[_token_str(token)] = 2**8 + i
            merges.append(_token_str(token[:-1]) + " " + _token_str(token[-1:]))
    return {
        "version": "1.0",
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
        "added_tokens": [
            {
                "id": 2**8 + len(merges),
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": True,
                "special": True,
            }
        ],
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


def make_encoding(tokenizer_json: dict) -> tiktoken.Encoding:
    return tiktoken.Encoding(name="hf_test", **hf_tokenizer_to_encoding_args(tokenizer_json))


def test_simple():
    enc = make_encoding(make_tokenizer_json())

    assert enc.encode("hello") == [258]
    assert enc.encode(" hello world") == [259, 260]
    assert enc.encode("hi there") == [104, 105, 32, 116, 104, 101, 114, 101]
    assert enc.decode([258]) == "hello"

    # Added tokens become special tokens, like <|endoftext|> for gpt2.
    assert enc.encode("<|endoftext|>", allowed_special="all") == [261]
    with pytest.raises(ValueError):
        enc.encode("<|endoftext|>")
    assert enc.encode("<|endoftext|>", disallowed_special=()) != [261]


def test_roundtrip():
    enc = make_encoding(make_tokenizer_json())

    for text in ["", "hello", " hello", " hello world", "hello  world", "abcdef", "日本語"]:
        assert enc.decode(enc.encode(text)) == text


def test_matches_gpt2():
    # gpt2's tokenizer.json is a byte level BPE with the same vocabulary and pattern as the
    # built in encoding, so converting it must reproduce tiktoken's gpt2 encoding exactly. The
    # merges are recovered from the ranks rather than hard coded, which keeps the test small.
    gpt2 = tiktoken.get_encoding("gpt2")
    ranks = gpt2._mergeable_ranks

    merges = []
    for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
        if len(token) < 2:
            continue
        for i in range(1, len(token)):
            first, second = token[:i], token[i:]
            if first in ranks and second in ranks and ranks[first] < rank and ranks[second] < rank:
                merges.append(_token_str(first) + " " + _token_str(second))
                break
        else:
            pytest.fail(f"Could not recover the merge for token {token!r}")

    tokenizer_json = make_tokenizer_json(merges)
    tokenizer_json["model"]["vocab"] = {_token_str(token): rank for token, rank in ranks.items()}
    tokenizer_json["added_tokens"][0]["id"] = gpt2._special_tokens["<|endoftext|>"]

    enc = make_encoding(tokenizer_json)
    assert enc._pat_str == gpt2._pat_str
    assert enc._mergeable_ranks == gpt2._mergeable_ranks
    assert enc._special_tokens == gpt2._special_tokens

    for text in [
        "hello world",
        "Hello world!",
        " hello",
        "hello  ",
        "0 00 000 0000",
        "today\n \n",
        "'s 't 'll 've 're 'd 'm",
        "请考试我的软件！12345",
        "👍",
        "def foo(x):\n    return x + 1\n",
    ]:
        assert enc.encode(text) == gpt2.encode(text)


def test_load_hf_tokenizer(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(make_tokenizer_json()))

    enc = load_hf_tokenizer(str(path))
    assert enc.name == "tokenizer.json"
    assert enc.encode("hello") == [258]

    enc = load_hf_tokenizer(str(path), name="custom")
    assert enc.name == "custom"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda tj: tj.__setitem__("normalizer", {"type": "NFC"}), "normalizer"),
        (
            lambda tj: tj.__setitem__("pre_tokenizer", {"type": "Metaspace"}),
            "pretokenizer",
        ),
        (lambda tj: tj.__setitem__("decoder", None), "byte level decoder"),
        (lambda tj: tj["model"].__setitem__("byte_fallback", True), "byte fallback"),
        (lambda tj: tj["model"].__setitem__("ignore_merges", True), "ignores merges"),
        (
            lambda tj: tj["model"].__setitem__(
                "vocab", {k: v for k, v in tj["model"]["vocab"].items() if v != 11}
            ),
            "missing single byte tokens",
        ),
        (
            lambda tj: tj["pre_tokenizer"].__setitem__("add_prefix_space", True),
            "prefix space",
        ),
        (
            lambda tj: tj["model"].__setitem__("merges", ["a b", "c d"]),
            "cannot represent",
        ),
    ],
)
def test_unsupported(mutate, message):
    tokenizer_json = make_tokenizer_json()
    mutate(tokenizer_json)
    with pytest.raises(ValueError, match=message):
        hf_tokenizer_to_encoding_args(tokenizer_json)


def test_custom_split_pattern():
    tokenizer_json = make_tokenizer_json()
    tokenizer_json["pre_tokenizer"] = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": "\\w+|."},
                "behavior": "Isolated",
                "invert": False,
            },
            {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
        ],
    }
    assert hf_tokenizer_to_encoding_args(tokenizer_json)["pat_str"] == "\\w+|."


def test_merge_order_must_match_token_ids():
    tokenizer_json = make_tokenizer_json()
    vocab = tokenizer_json["model"]["vocab"]
    # Give "ab" a higher token id than "abc" even though it merges first.
    vocab[_token_str(b"ab")], vocab[_token_str(b"abc")] = (
        vocab[_token_str(b"abc")],
        vocab[_token_str(b"ab")],
    )
    with pytest.raises(ValueError, match="Merge order"):
        hf_tokenizer_to_encoding_args(tokenizer_json)


def test_non_special_added_token_overlapping_vocab():
    tokenizer_json = make_tokenizer_json()
    tokenizer_json["added_tokens"].append(
        {
            "id": 2**8 + 5,
            "content": "hello",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": True,
            "special": False,
        }
    )
    with pytest.raises(ValueError, match="non special added token"):
        hf_tokenizer_to_encoding_args(tokenizer_json)


def test_does_not_mutate_input():
    tokenizer_json = make_tokenizer_json()
    original = copy.deepcopy(tokenizer_json)
    hf_tokenizer_to_encoding_args(tokenizer_json)
    assert tokenizer_json == original
