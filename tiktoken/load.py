from __future__ import annotations

import base64
import hashlib
import os


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
    lines = vocab_bpe_contents.split("\n")
    # the original GPT-2 vocab.bpe starts with a "#version: ..." header line; other tokenizers
    # (e.g. Qwen) ship a headerless file, so only skip the first line if it's a comment
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    bpe_merges = [tuple(merge_str.split()) for merge_str in lines if merge_str.strip()]

    def decode_data_gym(value: str) -> bytes:
        return bytes(data_gym_byte_to_byte[b] for b in value)

    # add the single byte tokens
    # if clobber_one_byte_tokens is True, we'll replace these with ones from the encoder json
    bpe_ranks = {bytes([b]): i for i, b in enumerate(rank_to_intbyte)}
    del rank_to_intbyte

    n = len(bpe_ranks)
    mergeable_tokens = []
    for first, second in bpe_merges:
        token = decode_data_gym(first) + decode_data_gym(second)
        bpe_ranks[token] = n
        n += 1
        mergeable_tokens.append(token)

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

    if bpe_ranks != encoder_json_loaded:
        # Some tokenizers (e.g. Qwen3.5) ship an encoder.json whose token ids do not match the
        # ranks implied by the merge order: the encoder assigns ids to tokens in blocks and
        # leaves gaps of unused tokens that no merge produces. Since the encoder's ids are what
        # the model's embedding table is indexed by, the encoder file is authoritative; keep it
        # as-is, and use the merges only to learn which tokens exist and their relative order.
        # The ids of the tokens produced by the merges must be strictly increasing in merge
        # order, otherwise the encoder and the merges are inconsistent and bpe breaks.
        if set(bpe_ranks) - set(encoder_json_loaded):
            raise ValueError(
                f"Encoder file {encoder_json_file} is missing tokens produced by the merges"
            )
        # the ids of the merged tokens must be strictly increasing in merge order, otherwise
        # the encoder and the merges are inconsistent and the bpe ranks would be wrong
        merged_ranks = [encoder_json_loaded[t] for t in mergeable_tokens]
        if len(set(merged_ranks)) != len(merged_ranks) or merged_ranks != sorted(merged_ranks):
            raise ValueError(f"Encoder file {encoder_json_file} is not consistent with the merges")
        # the encoder may contain extra tokens that no merge produces (e.g. unused tokens
        # reserved during training); keep them, since their ids index the embedding table
        bpe_ranks = dict(encoder_json_loaded)

    return bpe_ranks


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
