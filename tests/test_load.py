import json

import pytest

from tiktoken.load import data_gym_to_mergeable_bpe_ranks


def _byte_encoder():
    # replicate the data-gym byte <-> unicode mapping used by the loader;
    # single byte tokens are assigned ranks in this order
    rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]
    data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
    n = 0
    for b in range(2**8):
        if b not in rank_to_intbyte:
            data_gym_byte_to_byte[chr(2**8 + n)] = b
            n += 1
    return {c: i for i, c in enumerate(data_gym_byte_to_byte)}


def _write_files(tmp_path, merges: str, encoder: dict):
    merges_file = tmp_path / "merges.txt"
    merges_file.write_text(merges, encoding="utf-8")
    encoder_file = tmp_path / "encoder.json"
    encoder_file.write_text(json.dumps(encoder), encoding="utf-8")
    return str(merges_file), str(encoder_file)


MERGES = "Ġ Ġ\ni n\n"  # no "#version:" header, like Qwen's merges.txt


def test_merges_without_header(tmp_path):
    encoder = _byte_encoder()
    encoder["ĠĠ"] = 256
    encoder["in"] = 257
    merges_file, encoder_file = _write_files(tmp_path, MERGES, encoder)
    ranks = data_gym_to_mergeable_bpe_ranks(merges_file, encoder_file)
    assert ranks[b"  "] == 256
    assert ranks[b"in"] == 257
    assert len(ranks) == 258


def test_merges_with_header(tmp_path):
    encoder = _byte_encoder()
    encoder["ĠĠ"] = 256
    encoder["in"] = 257
    merges_file, encoder_file = _write_files(tmp_path, "#version: 0.2\n" + MERGES, encoder)
    ranks = data_gym_to_mergeable_bpe_ranks(merges_file, encoder_file)
    assert ranks[b"  "] == 256
    assert ranks[b"in"] == 257
    assert len(ranks) == 258


def test_encoder_with_unused_tokens(tmp_path):
    # some vocabs (e.g. Qwen3.5) contain tokens that no merge produces, interleaved
    # in the id space; the encoder is authoritative for their ids
    encoder = _byte_encoder()
    encoder["ĠĠ"] = 256
    encoder["extra"] = 257  # reserved token, not produced by any merge
    encoder["in"] = 258
    merges_file, encoder_file = _write_files(tmp_path, MERGES, encoder)
    ranks = data_gym_to_mergeable_bpe_ranks(merges_file, encoder_file)
    assert ranks[b"  "] == 256
    assert ranks[b"extra"] == 257
    assert ranks[b"in"] == 258
    assert len(ranks) == 259


def test_encoder_missing_merge_token_is_rejected(tmp_path):
    encoder = _byte_encoder()
    encoder["ĠĠ"] = 256
    # "in" is produced by a merge but missing from the encoder
    merges_file, encoder_file = _write_files(tmp_path, MERGES, encoder)
    with pytest.raises(ValueError):
        data_gym_to_mergeable_bpe_ranks(merges_file, encoder_file)


def test_encoder_inconsistent_with_merge_order_is_rejected(tmp_path):
    encoder = _byte_encoder()
    # swapped ids of the two merged tokens: not consistent with the merge order
    encoder["ĠĠ"] = 257
    encoder["in"] = 256
    merges_file, encoder_file = _write_files(tmp_path, MERGES, encoder)
    with pytest.raises(ValueError):
        data_gym_to_mergeable_bpe_ranks(merges_file, encoder_file)
