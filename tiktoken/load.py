from __future__ import annotations

import base64
import hashlib
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tiktoken.core import Encoding


def read_file(blobpath: str) -> bytes:
    if "://" not in blobpath:
        with open(blobpath, "rb", buffering=0) as f:
            return f.read()

    if blobpath.startswith(("http://", "https://")):
        # avoiding blobfile for public files helps avoid auth issues, like MFA prompts.
        import requests

        resp = requests.get(blobpath)
        resp.raise_for_status()
        return resp.content

    try:
        import blobfile
    except ImportError as e:
        raise ImportError(
            "blobfile is not installed. Please install it by running `pip install blobfile`."
        ) from e
    return blobfile.read_bytes(blobpath)


def check_hash(data: bytes, expected_hash: str) -> bool:
    actual_hash = hashlib.sha256(data).hexdigest()
    return actual_hash == expected_hash


def read_file_cached(blobpath: str, expected_hash: str | None = None) -> bytes:
    user_specified_cache = True
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        import tempfile

        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
        user_specified_cache = False

    if cache_dir == "":
        # disable caching
        return read_file(blobpath)

    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()

    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        with open(cache_path, "rb", buffering=0) as f:
            data = f.read()
        if expected_hash is None or check_hash(data, expected_hash):
            return data

        # the cached file does not match the hash, remove it and re-fetch
        try:
            os.remove(cache_path)
        except OSError:
            pass

    contents = read_file(blobpath)
    if expected_hash and not check_hash(contents, expected_hash):
        raise ValueError(
            f"Hash mismatch for data downloaded from {blobpath} (expected {expected_hash}). "
            f"This may indicate a corrupted download. Please try again."
        )

    import uuid

    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_filename = cache_path + "." + str(uuid.uuid4()) + ".tmp"
        with open(tmp_filename, "wb") as f:
            f.write(contents)
        os.rename(tmp_filename, cache_path)
    except OSError:
        # don't raise if we can't write to the default cache, e.g. issue #75
        if user_specified_cache:
            raise

    return contents


def data_gym_to_mergeable_bpe_ranks(
    vocab_bpe_file: str,
    encoder_json_file: str,
    vocab_bpe_hash: str | None = None,
    encoder_json_hash: str | None = None,
    clobber_one_byte_tokens: bool = False,
) -> dict[bytes, int]:
    # NB: do not add caching to this function
    rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]

    data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
    n = 0
    for b in range(2**8):
        if b not in rank_to_intbyte:
            rank_to_intbyte.append(b)
            data_gym_byte_to_byte[chr(2**8 + n)] = b
            n += 1
    assert len(rank_to_intbyte) == 2**8

    # vocab_bpe contains the merges along with associated ranks
    vocab_bpe_contents = read_file_cached(vocab_bpe_file, vocab_bpe_hash).decode()
    bpe_merges = [tuple(merge_str.split()) for merge_str in vocab_bpe_contents.split("\n")[1:-1]]

    def decode_data_gym(value: str) -> bytes:
        return bytes(data_gym_byte_to_byte[b] for b in value)

    # add the single byte tokens
    # if clobber_one_byte_tokens is True, we'll replace these with ones from the encoder json
    bpe_ranks = {bytes([b]): i for i, b in enumerate(rank_to_intbyte)}
    del rank_to_intbyte

    # add the merged tokens
    n = len(bpe_ranks)
    for first, second in bpe_merges:
        bpe_ranks[decode_data_gym(first) + decode_data_gym(second)] = n
        n += 1

    import json

    # check that the encoder file matches the merges file
    # this sanity check is important since tiktoken assumes that ranks are ordered the same
    # as merge priority
    encoder_json = json.loads(read_file_cached(encoder_json_file, encoder_json_hash))
    encoder_json_loaded = {decode_data_gym(k): v for k, v in encoder_json.items()}
    # drop these two special tokens if present, since they're not mergeable bpe tokens
    encoder_json_loaded.pop(b"<|endoftext|>", None)
    encoder_json_loaded.pop(b"<|startoftext|>", None)

    if clobber_one_byte_tokens:
        for k in encoder_json_loaded:
            if len(k) == 1:
                bpe_ranks[k] = encoder_json_loaded[k]

    assert bpe_ranks == encoder_json_loaded

    return bpe_ranks


# HuggingFace's ByteLevel pretokeniser splits with the GPT-2 pattern when `use_regex` is set. This
# is how tiktoken spells that pattern; the two formulations produce the same splits.
_HF_BYTE_LEVEL_PAT_STR = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)

# Bytes that never occur in valid UTF-8, so a tokeniser does not need single byte tokens for them.
# tiktoken only ever sees UTF-8 encoded text, so it does not need them either.
_INVALID_UTF8_BYTES = frozenset([*range(0xC0, 0xC2), *range(0xF5, 0x100)])


def _bytes_to_unicode() -> dict[int, str]:
    """Returns the byte to unicode mapping used by byte level BPE tokenisers."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _hf_token_bytes(token: str, byte_decoder: dict[str, int]) -> bytes:
    try:
        return bytes(byte_decoder[c] for c in token)
    except KeyError as e:
        raise ValueError(
            f"Token {token!r} is not encoded with the byte level alphabet that tiktoken supports"
        ) from e


def _hf_pat_str(tokenizer_json: dict[str, Any]) -> str:
    """Returns the tiktoken equivalent of a HuggingFace pretokeniser.

    Only pretokenisers that tiktoken can reproduce are supported. Anything that rewrites the text
    (e.g. SentencePiece's Metaspace) would need a regex that tiktoken cannot express, so we reject
    it rather than silently producing different tokens.
    """
    pretokenizer = tokenizer_json.get("pre_tokenizer")
    if pretokenizer is None:
        raise ValueError("HuggingFace tokenizer has no pretokenizer, which tiktoken cannot support")
    if pretokenizer.get("type") == "ByteLevel":
        if pretokenizer.get("add_prefix_space"):
            raise ValueError(
                "HuggingFace tokenizer adds a prefix space, which tiktoken cannot support"
            )
        if not pretokenizer.get("use_regex", True):
            raise ValueError(
                "HuggingFace tokenizer does not split on the GPT-2 pattern, which tiktoken "
                "cannot support"
            )
        return _HF_BYTE_LEVEL_PAT_STR
    if pretokenizer.get("type") != "Sequence":
        raise ValueError(
            f"HuggingFace tokenizer uses a {pretokenizer.get('type')!r} pretokenizer, "
            "which tiktoken cannot support"
        )
    parts = pretokenizer.get("pretokenizers", [])
    if len(parts) != 2 or parts[0].get("type") != "Split" or parts[1].get("type") != "ByteLevel":
        raise ValueError(
            "HuggingFace tokenizer uses an unsupported pretokenizer, which tiktoken cannot support"
        )
    split, byte_level = parts
    if split.get("behavior", "Isolated") != "Isolated" or split.get("invert"):
        raise ValueError(
            "HuggingFace tokenizer uses an unsupported split behaviour, which tiktoken cannot "
            "support"
        )
    if byte_level.get("add_prefix_space") or byte_level.get("use_regex", True):
        raise ValueError(
            "HuggingFace tokenizer uses an unsupported byte level pretokenizer, which tiktoken "
            "cannot support"
        )
    pattern = split.get("pattern")
    if not isinstance(pattern, dict) or "Regex" not in pattern:
        raise ValueError(
            "HuggingFace tokenizer uses an unsupported split pattern, which tiktoken cannot support"
        )
    return pattern["Regex"]


def hf_tokenizer_to_encoding_args(tokenizer_json: dict[str, Any]) -> dict[str, Any]:
    """Converts a HuggingFace `tokenizer.json` to arguments for `tiktoken.Encoding`.

    This is a dependency free equivalent of what libraries such as AutoTikTokenizer do, so that
    byte level BPE tokenizers from the HuggingFace Hub can be used with tiktoken's fast
    implementation. See https://github.com/openai/tiktoken/issues/358.

    The token ids in `tokenizer.json` are the ranks that tiktoken expects, so they are read
    straight from the vocabulary. The merge order is checked to agree with them, since tiktoken
    assumes that ranks are ordered the same as merge priority.

    Added tokens are exposed as special tokens. tiktoken only matches those when they are allowed
    explicitly, so use `encode(..., allowed_special="all")` to reproduce how HuggingFace matches
    them.

    Tokenizers that need text rewriting (normalizers, SentencePiece's Metaspace, ...) are rejected
    with a `ValueError` rather than silently producing different tokens, since tiktoken cannot
    express those transformations.
    """
    pat_str = _hf_pat_str(tokenizer_json)
    normalizer = tokenizer_json.get("normalizer")
    if normalizer is not None:
        raise ValueError(
            "HuggingFace tokenizer has a normalizer, which tiktoken cannot support. "
            "Normalize the text before encoding it instead."
        )
    decoder = tokenizer_json.get("decoder") or {}
    if decoder.get("type") != "ByteLevel":
        raise ValueError(
            "HuggingFace tokenizer does not use a byte level decoder, which tiktoken cannot support"
        )

    model = tokenizer_json.get("model") or {}
    if model.get("type", "BPE") != "BPE" or not model.get("merges"):
        raise ValueError(
            "HuggingFace tokenizer is not a BPE tokenizer, which tiktoken cannot support"
        )
    if model.get("byte_fallback"):
        raise ValueError("HuggingFace tokenizer uses byte fallback, which tiktoken cannot support")
    if model.get("ignore_merges"):
        raise ValueError("HuggingFace tokenizer ignores merges, which tiktoken cannot support")

    vocab = model["vocab"]
    byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}

    added_tokens = tokenizer_json.get("added_tokens", [])
    special_tokens = {token["content"]: token["id"] for token in added_tokens}
    # tiktoken only matches special tokens when they are explicitly allowed, whereas HuggingFace
    # matches non special added tokens greedily while encoding ordinary text. An added token that
    # overlaps the vocabulary (OLMo's runs of spaces, for instance) would therefore encode
    # differently, so reject it rather than silently producing different tokens.
    for token in added_tokens:
        if not token.get("special") and token["content"] in vocab:
            raise ValueError(
                f"HuggingFace tokenizer has a non special added token {token['content']!r} that "
                "overlaps the vocabulary. tiktoken cannot reproduce how HuggingFace matches it."
            )

    mergeable_ranks: dict[bytes, int] = {}
    for token, token_id in vocab.items():
        if token in special_tokens:
            continue
        token_bytes = _hf_token_bytes(token, byte_decoder)
        if len(token_bytes) == 1:
            mergeable_ranks[token_bytes] = token_id

    missing_bytes = [
        b for b in range(2**8) if b not in _INVALID_UTF8_BYTES and bytes([b]) not in mergeable_ranks
    ]
    if missing_bytes:
        raise ValueError(
            "HuggingFace tokenizer is missing single byte tokens, which tiktoken requires. "
            f"Missing bytes: {missing_bytes[:10]}"
        )

    # Token ids must increase along with merge priority, otherwise the vocabulary does not agree
    # with the merge order and tiktoken would merge in the wrong order.
    seen_tokens: set[str] = set()
    previous_token_id = -1
    for merge in model["merges"]:
        parts = merge.split(" ") if isinstance(merge, str) else merge
        if len(parts) != 2:
            raise ValueError(f"Malformed merge {merge!r} in HuggingFace tokenizer")
        merged_token = parts[0] + parts[1]
        if merged_token in seen_tokens or merged_token in special_tokens:
            continue
        if merged_token not in vocab:
            continue
        seen_tokens.add(merged_token)
        token_id = vocab[merged_token]
        if token_id <= previous_token_id:
            raise ValueError(
                f"Merge order does not match token ids for token {merged_token!r}. "
                "tiktoken assumes that ranks are ordered the same as merge priority."
            )
        previous_token_id = token_id
        token_bytes = _hf_token_bytes(merged_token, byte_decoder)
        if token_bytes not in mergeable_ranks:
            mergeable_ranks[token_bytes] = token_id

    # Token ids must be unique, otherwise decoding is ambiguous.
    collisions = set(special_tokens.values()) & set(mergeable_ranks.values())
    if collisions:
        raise ValueError(
            f"HuggingFace tokenizer reuses token ids for added tokens: {sorted(collisions)[:10]}"
        )

    # Every token must be reachable, otherwise the encoding would be missing tokens that the
    # original tokenizer can produce.
    unreachable = []
    for token in vocab:
        if token in special_tokens:
            continue
        try:
            token_bytes = _hf_token_bytes(token, byte_decoder)
        except ValueError:
            token_bytes = None
        if token_bytes not in mergeable_ranks:
            unreachable.append(token)
    if unreachable:
        raise ValueError(
            f"HuggingFace tokenizer has tokens that tiktoken cannot represent: {unreachable[:10]}"
        )

    return {
        "pat_str": pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }


def load_hf_tokenizer(
    tokenizer_json_file: str, *, name: str | None = None, expected_hash: str | None = None
) -> "Encoding":
    """Loads a HuggingFace `tokenizer.json` file as a `tiktoken.Encoding`.

    See `hf_tokenizer_to_encoding_args` for the formats that are supported.
    """
    import json

    import tiktoken

    contents = read_file_cached(tokenizer_json_file, expected_hash)
    encoding_args = hf_tokenizer_to_encoding_args(json.loads(contents))
    if name is None:
        name = os.path.basename(tokenizer_json_file)
    return tiktoken.Encoding(name=name, **encoding_args)


def dump_tiktoken_bpe(bpe_ranks: dict[bytes, int], tiktoken_bpe_file: str) -> None:
    try:
        import blobfile
    except ImportError as e:
        raise ImportError(
            "blobfile is not installed. Please install it by running `pip install blobfile`."
        ) from e
    with blobfile.BlobFile(tiktoken_bpe_file, "wb") as f:
        for token, rank in sorted(bpe_ranks.items(), key=lambda x: x[1]):
            f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")


def load_tiktoken_bpe(tiktoken_bpe_file: str, expected_hash: str | None = None) -> dict[bytes, int]:
    # NB: do not add caching to this function
    contents = read_file_cached(tiktoken_bpe_file, expected_hash)
    ret = {}
    for line in contents.splitlines():
        if not line:
            continue
        try:
            token, rank = line.split()
            ret[base64.b64decode(token)] = int(rank)
        except Exception as e:
            raise ValueError(f"Error parsing line {line!r} in {tiktoken_bpe_file}") from e
    return ret
