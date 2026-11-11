"""Tests for tokenising scripts that use combining marks, e.g. Bengali (issue #593).

The legacy split patterns only match ``\\p{L}``. Combining marks (``\\p{M}``) therefore always
start a new piece, so a word like "বাংলা" is split into "ব", "াং", "ল", "া" and BPE can never
merge a letter with the vowel sign that follows it. ``multilingual_pat_str`` matches both
categories together, which keeps such words in a single piece.
"""

import unicodedata

import regex
import pytest

import tiktoken
from tiktoken._educational import bpe_train
from tiktoken.core import Encoding
from tiktoken_ext.openai_public import multilingual_pat_str, r50k_pat_str

# "I sing in Bengali. My Bengali language."
BENGALI = "আমি বাংলায় গান গাই। আমার বাংলা ভাষা।"

# A small corpus with the words from the issue, repeated so there is something to merge.
BENGALI_CORPUS = " ".join(
    "আমি বাংলায় গান গাই আমার বাংলা ভাষা বাংলাদেশের রাজধানী ঢাকা নাশতা গোসল চাচা".split() * 60
)


_COMBINING_MARK_CATEGORIES = {"Mn", "Mc", "Me"}


def _spans_letter_and_mark(token: bytes) -> bool:
    """Whether a token contains both a letter and a combining mark."""
    try:
        text = token.decode()
    except UnicodeDecodeError:
        return False
    categories = [unicodedata.category(c) for c in text]
    has_letter = any(c.startswith("L") for c in categories)
    has_mark = any(c in _COMBINING_MARK_CATEGORIES for c in categories)
    return has_letter and has_mark


def _common_prefix_len(a: str, b: str) -> int:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _token_offsets_reference(enc: Encoding, tokens: list[int]) -> list[int]:
    text = enc.decode(tokens, errors="strict")
    return [
        _common_prefix_len(text, enc.decode(tokens[:i], errors="ignore"))
        for i in range(len(tokens))
    ]


def test_legacy_pattern_splits_bengali_at_combining_marks():
    # This is the defect from the issue: the split rule breaks words at every vowel sign.
    pattern = regex.compile(r50k_pat_str)
    assert regex.findall(pattern, "বাংলা") == ["ব", "াং", "ল", "া"]
    assert regex.findall(pattern, "আমি") == ["আম", "ি"]


@pytest.mark.parametrize(
    "text",
    [
        "বাংলা",  # Bengali: base letters + vowel signs
        "আমি বাংলায় গান গাই",  # Bengali phrase
        "नमस्ते",  # Devanagari
        "สวัสดี",  # Thai
        "שָׁלוֹם",  # Hebrew with points
    ],
)
def test_multilingual_pattern_keeps_words_whole(text):
    pattern = regex.compile(multilingual_pat_str)
    pieces = regex.findall(pattern, text)
    # Splitting is lossless and every piece is a whole word (possibly with its leading space),
    # i.e. no piece boundary falls inside a word.
    assert "".join(pieces) == text
    assert [piece.strip() for piece in pieces] == text.split(" ")


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "hello  world",
        "  leading",
        "trailing  ",
        "中文字符",
        "a1b2c3",
        "'s 't 're 've 'm 'll 'd",
        "don't",
        "line\nbreaks\n\n",
        "😀👍",
    ],
)
def test_multilingual_pattern_matches_legacy_for_text_without_marks(text):
    # Text that contains no combining marks must be split exactly as before, so switching
    # patterns cannot change token counts for the scripts the legacy patterns already handled.
    legacy = regex.findall(regex.compile(r50k_pat_str), text)
    multilingual = regex.findall(regex.compile(multilingual_pat_str), text)
    assert legacy == multilingual


@pytest.mark.parametrize("encoding_name", ["gpt2", "r50k_base", "cl100k_base", "o200k_base"])
def test_bengali_roundtrip(encoding_name):
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(BENGALI)
    assert enc.decode(tokens) == BENGALI
    assert enc.decode(enc.encode_ordinary(BENGALI)) == BENGALI

    # Decoding bytes must be lossless too, even though Bengali characters are multi-byte.
    assert enc.decode_bytes(tokens) == BENGALI.encode()

    # The Rust and Python regex split paths must agree.
    assert enc._encode_only_native_bpe(BENGALI) == tokens


@pytest.mark.parametrize("encoding_name", ["r50k_base", "cl100k_base", "o200k_base"])
def test_bengali_offsets(encoding_name):
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(BENGALI)
    text, offsets = enc.decode_with_offsets(tokens)
    assert text == BENGALI
    assert offsets == _token_offsets_reference(enc, tokens)
    assert offsets == sorted(offsets)


def test_o200k_tokenises_bengali_cheaply():
    # o200k_base already uses a mark-aware split rule and has Bengali vocabulary entries, so it
    # is the encoding to use for Bengali today. cl100k_base has no Bengali tokens at all.
    o200k = tiktoken.get_encoding("o200k_base")
    cl100k = tiktoken.get_encoding("cl100k_base")

    assert len(o200k.encode(BENGALI)) * 2 < len(cl100k.encode(BENGALI))

    bengali_tokens = [
        token
        for token in cl100k._mergeable_ranks
        if any(0x0980 <= byte <= 0x09FF for byte in token)
    ]
    assert bengali_tokens == []


def test_multilingual_pattern_unlocks_bengali_subwords():
    # Train two toy vocabularies on the same Bengali corpus, one with each split rule. The legacy
    # rule makes letter+mark merges impossible, so the vocabulary ends up with none of them and
    # the text costs far more tokens.
    legacy_ranks = bpe_train(BENGALI_CORPUS, vocab_size=600, pat_str=r50k_pat_str, visualise=None)
    multi_ranks = bpe_train(
        BENGALI_CORPUS, vocab_size=600, pat_str=multilingual_pat_str, visualise=None
    )

    assert not any(_spans_letter_and_mark(token) for token in legacy_ranks)
    assert any(_spans_letter_and_mark(token) for token in multi_ranks)

    legacy_enc = Encoding(
        name="legacy", pat_str=r50k_pat_str, mergeable_ranks=legacy_ranks, special_tokens={}
    )
    multi_enc = Encoding(
        name="multilingual",
        pat_str=multilingual_pat_str,
        mergeable_ranks=multi_ranks,
        special_tokens={},
    )

    for text in ["আমি বাংলায় গান গাই।", "আমার বাংলা ভাষা।", "নাশতা গোসল চাচা"]:
        legacy_tokens = legacy_enc.encode(text)
        multi_tokens = multi_enc.encode(text)
        assert legacy_enc.decode(legacy_tokens) == text
        assert multi_enc.decode(multi_tokens) == text
        assert len(multi_tokens) < len(legacy_tokens)


def test_bpe_train_stops_when_corpus_is_exhausted():
    # A low-resource corpus has fewer mergeable pairs than vocab_size asks for. Training used to
    # raise `ValueError: max() arg is an empty sequence`; it should return what it learned.
    ranks = bpe_train("আমি", vocab_size=1000, pat_str=multilingual_pat_str, visualise=None)

    assert sorted(ranks.values()) == list(range(len(ranks)))
    assert len(ranks) > 256
    assert len(ranks) < 1000

    enc = Encoding(
        name="tiny", pat_str=multilingual_pat_str, mergeable_ranks=ranks, special_tokens={}
    )
    assert enc.decode(enc.encode("আমি")) == "আমি"
