# Note that there are more actual tests, they're just not currently public :-)

from typing import Callable

import hypothesis
import hypothesis.strategies as st
import pytest

import tiktoken

from .test_helpers import ENCODING_FACTORIES, MAX_EXAMPLES


def test_simple():
    enc = tiktoken.get_encoding("gpt2")
    assert enc.encode("hello world") == [31373, 995]
    assert enc.decode([31373, 995]) == "hello world"
    assert enc.encode("hello <|endoftext|>", allowed_special="all") == [31373, 220, 50256]

    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.encode("hello world") == [15339, 1917]
    assert enc.decode([15339, 1917]) == "hello world"
    assert enc.encode("hello <|endoftext|>", allowed_special="all") == [15339, 220, 100257]

    for enc_name in tiktoken.list_encoding_names():
        enc = tiktoken.get_encoding(enc_name)
        for token in range(min(10_000, enc.max_token_value - 1)):
            assert enc.encode_single_token(enc.decode_single_token_bytes(token)) == token


def test_simple_repeated():
    enc = tiktoken.get_encoding("gpt2")
    assert enc.encode("0") == [15]
    assert enc.encode("00") == [405]
    assert enc.encode("000") == [830]
    assert enc.encode("0000") == [2388]
    assert enc.encode("00000") == [20483]
    assert enc.encode("000000") == [10535]
    assert enc.encode("0000000") == [24598]
    assert enc.encode("00000000") == [8269]
    assert enc.encode("000000000") == [10535, 830]
    assert enc.encode("0000000000") == [8269, 405]
    assert enc.encode("00000000000") == [8269, 830]
    assert enc.encode("000000000000") == [8269, 2388]
    assert enc.encode("0000000000000") == [8269, 20483]
    assert enc.encode("00000000000000") == [8269, 10535]
    assert enc.encode("000000000000000") == [8269, 24598]
    assert enc.encode("0000000000000000") == [25645]
    assert enc.encode("00000000000000000") == [8269, 10535, 830]


def test_large_repeated():
    enc = tiktoken.get_encoding("o200k_base")

    # Large inputs should be handled without raising.
    tokens = enc.encode("x" * 1_000_000)
    assert tokens


def test_simple_regex():
    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.encode("rer") == [38149]
    assert enc.encode("'rer") == [2351, 81]
    assert enc.encode("today\n ") == [31213, 198, 220]
    assert enc.encode("today\n \n") == [31213, 27907]
    assert enc.encode("today\n  \n") == [31213, 14211]


def test_basic_encode():
    enc = tiktoken.get_encoding("r50k_base")
    assert enc.encode("hello world") == [31373, 995]

    enc = tiktoken.get_encoding("p50k_base")
    assert enc.encode("hello world") == [31373, 995]

    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.encode("hello world") == [15339, 1917]
    assert enc.encode(" \x850") == [220, 126, 227, 15]


def test_encode_empty():
    enc = tiktoken.get_encoding("r50k_base")
    assert enc.encode("") == []


def test_encode_bytes():
    enc = tiktoken.get_encoding("cl100k_base")
    assert enc._encode_bytes(b" \xec\x8b\xa4\xed") == [62085]
    for i in range(10):
        bytestring = b"\x80" * i
        assert enc.decode_bytes(enc._encode_bytes(bytestring)) == bytestring


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
@hypothesis.given(bytestring=st.binary())
@hypothesis.settings(deadline=None, max_examples=MAX_EXAMPLES)
def test_hyp_encode_bytes(make_enc: Callable[[], tiktoken.Encoding], bytestring: bytes):
    enc = make_enc()
    assert enc.decode_bytes(enc._encode_bytes(bytestring)) == bytestring


def test_encode_surrogate_pairs():
    enc = tiktoken.get_encoding("cl100k_base")

    assert enc.encode("👍") == [9468, 239, 235]
    # surrogate pair gets converted to codepoint
    assert enc.encode("\ud83d\udc4d") == [9468, 239, 235]

    # lone surrogate just gets replaced
    assert enc.encode("\ud83d") == enc.encode("�")


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
def test_catastrophically_repetitive(make_enc: Callable[[], tiktoken.Encoding]):
    enc = make_enc()
    for c in ["^", "0", "a", "'s", " ", "\n"]:
        big_value = c * 10_000
        assert big_value == enc.decode(enc.encode(big_value))

        big_value = " " + big_value
        assert big_value == enc.decode(enc.encode(big_value))

        big_value = big_value + "\n"
        assert big_value == enc.decode(enc.encode(big_value))


# ====================
# Roundtrip
# ====================


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
def test_basic_roundtrip(make_enc):
    enc = make_enc()
    for value in (
        "hello",
        "hello ",
        "hello  ",
        " hello",
        " hello ",
        " hello  ",
        "hello world",
        "请考试我的软件！12345",
    ):
        assert value == enc.decode(enc.encode(value))
        assert value == enc.decode(enc.encode_ordinary(value))


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
@hypothesis.given(text=st.text())
@hypothesis.settings(deadline=None, max_examples=MAX_EXAMPLES)
def test_hyp_roundtrip(make_enc: Callable[[], tiktoken.Encoding], text):
    enc = make_enc()

    # disallowed_special=() so that random text containing a special token string
    # is treated as text instead of raising
    assert text == enc.decode(enc.encode(text, disallowed_special=()))


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
def test_single_token_roundtrip(make_enc: Callable[[], tiktoken.Encoding]):
    enc = make_enc()

    for token in range(enc.n_vocab):
        try:
            token_bytes = enc.decode_single_token_bytes(token)
        except KeyError:
            continue
        assert enc.encode_single_token(token_bytes) == token


# ====================
# Special tokens
# ====================


def test_special_token():
    enc = tiktoken.get_encoding("cl100k_base")

    eot = enc.encode_single_token("<|endoftext|>")
    assert eot == enc.eot_token
    fip = enc.encode_single_token("<|fim_prefix|>")
    fim = enc.encode_single_token("<|fim_middle|>")

    text = "<|endoftext|> hello <|fim_prefix|>"
    assert eot not in enc.encode(text, disallowed_special=())
    with pytest.raises(ValueError):
        enc.encode(text)
    with pytest.raises(ValueError):
        enc.encode(text, disallowed_special="all")
    with pytest.raises(ValueError):
        enc.encode(text, disallowed_special={"<|endoftext|>"})
    with pytest.raises(ValueError):
        enc.encode(text, disallowed_special={"<|fim_prefix|>"})

    text = "<|endoftext|> hello <|fim_prefix|> there <|fim_middle|>"
    tokens = enc.encode(text, disallowed_special=())
    assert eot not in tokens
    assert fip not in tokens
    assert fim not in tokens

    tokens = enc.encode(text, allowed_special="all", disallowed_special=())
    assert eot in tokens
    assert fip in tokens
    assert fim in tokens

    tokens = enc.encode(text, allowed_special="all", disallowed_special="all")
    assert eot in tokens
    assert fip in tokens
    assert fim in tokens

    tokens = enc.encode(text, allowed_special={"<|fim_prefix|>"}, disallowed_special=())
    assert eot not in tokens
    assert fip in tokens
    assert fim not in tokens

    tokens = enc.encode(text, allowed_special={"<|endoftext|>"}, disallowed_special=())
    assert eot in tokens
    assert fip not in tokens
    assert fim not in tokens

    tokens = enc.encode(text, allowed_special={"<|fim_middle|>"}, disallowed_special=())
    assert eot not in tokens
    assert fip not in tokens
    assert fim in tokens


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
@hypothesis.given(text=st.text())
@hypothesis.settings(deadline=None, max_examples=MAX_EXAMPLES)
def test_hyp_special_ordinary(make_enc, text: str):
    enc = make_enc()
    assert enc.encode_ordinary(text) == enc.encode(text, disallowed_special=())


def test_resolve_special_token_args():
    enc = tiktoken.get_encoding("cl100k_base")
    all_tokens = enc.special_tokens_set
    eot = enc.decode([enc.eot_token])

    # allowed_special="all" implies nothing is disallowed
    allowed, disallowed = enc._resolve_special_token_args("all", "all")
    assert allowed == all_tokens
    assert disallowed == frozenset()

    allowed, disallowed = enc._resolve_special_token_args("all", ())
    assert allowed == all_tokens
    assert disallowed == frozenset()

    # by default everything is disallowed
    allowed, disallowed = enc._resolve_special_token_args(set(), "all")
    assert allowed == frozenset()
    assert disallowed == all_tokens

    # disallowed defaults to everything not allowed
    allowed, disallowed = enc._resolve_special_token_args({eot}, "all")
    assert allowed == {eot}
    assert disallowed == all_tokens - {eot}

    # explicit disallowed sets are respected as-is
    allowed, disallowed = enc._resolve_special_token_args({"<|fim_prefix|>"}, {eot})
    assert allowed == {"<|fim_prefix|>"}
    assert disallowed == {eot}

    # non-frozenset collections are accepted for both arguments
    allowed, disallowed = enc._resolve_special_token_args(["<|fim_prefix|>"], [eot])
    assert allowed == {"<|fim_prefix|>"}
    assert disallowed == {eot}

    # the O(n) disallowed computation is cached across calls
    _, disallowed_first = enc._resolve_special_token_args({eot}, "all")
    _, disallowed_second = enc._resolve_special_token_args({eot}, "all")
    assert disallowed_first is disallowed_second


def test_many_special_tokens():
    # See #441: encodings can have a large special token library, of which only a
    # small subset is in use. Encoding should stay correct (and cheap) in that case.
    base = tiktoken.get_encoding("gpt2")
    n_special = 5000
    special_tokens = {
        f"<|extra_special_token_{i}|>": 50257 + i for i in range(n_special)
    }
    enc = tiktoken.Encoding(
        name="test_many_special_tokens",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )
    assert len(enc.special_tokens_set) == n_special

    allowed = {"<|extra_special_token_1|>", "<|extra_special_token_2|>"}

    text = "hello world"
    assert enc.encode(text) == base.encode(text)
    assert enc.encode(text, allowed_special="all") == [31373, 995]
    assert enc.encode_ordinary(text) == base.encode_ordinary(text)

    # a disallowed special token raises
    with pytest.raises(ValueError):
        enc.encode(text + "<|extra_special_token_3|>")

    # allowed special tokens are encoded as tokens, disallowed ones as text
    text = "a<|extra_special_token_1|>b<|extra_special_token_3|>c"
    tokens = enc.encode(text, allowed_special=allowed, disallowed_special=())
    assert enc.decode(tokens) == text
    assert enc.encode_single_token("<|extra_special_token_1|>") in tokens
    assert enc.encode_single_token("<|extra_special_token_3|>") not in tokens

    # all special tokens encoded as tokens
    tokens = enc.encode(text, allowed_special="all")
    assert enc.decode(tokens) == text
    assert enc.encode_single_token("<|extra_special_token_1|>") in tokens
    assert enc.encode_single_token("<|extra_special_token_3|>") in tokens


def test_no_special_tokens():
    # An encoding with no special tokens should behave like an ordinary encoding
    base = tiktoken.get_encoding("gpt2")
    enc = tiktoken.Encoding(
        name="test_no_special_tokens",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={},
    )
    for text in ("hello world", "", "<|endoftext|>"):
        assert enc.encode(text) == base.encode_ordinary(text)
        assert enc.encode_ordinary(text) == base.encode_ordinary(text)
        assert enc.decode(enc.encode(text)) == text


# ====================
# Batch encoding
# ====================


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
def test_batch_encode(make_enc: Callable[[], tiktoken.Encoding]):
    enc = make_enc()
    text1 = "hello world"
    text2 = "goodbye world"

    assert enc.encode_batch([text1]) == [enc.encode(text1)]
    assert enc.encode_batch([text1, text2]) == [enc.encode(text1), enc.encode(text2)]

    assert enc.encode_ordinary_batch([text1]) == [enc.encode_ordinary(text1)]
    assert enc.encode_ordinary_batch([text1, text2]) == [
        enc.encode_ordinary(text1),
        enc.encode_ordinary(text2),
    ]


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
@hypothesis.given(batch=st.lists(st.text()))
@hypothesis.settings(deadline=None, max_examples=MAX_EXAMPLES)
def test_hyp_batch_roundtrip(make_enc: Callable[[], tiktoken.Encoding], batch):
    enc = make_enc()

    encoded = enc.encode_batch(batch, allowed_special="all")
    assert encoded == [enc.encode(t, allowed_special="all") for t in batch]
    decoded = enc.decode_batch(encoded)
    assert decoded == batch
