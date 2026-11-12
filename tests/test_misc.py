import subprocess
import sys

import pytest

import tiktoken


def test_encoding_for_model():
    enc = tiktoken.encoding_for_model("gpt2")
    assert enc.name == "gpt2"
    enc = tiktoken.encoding_for_model("text-davinci-003")
    assert enc.name == "p50k_base"
    enc = tiktoken.encoding_for_model("text-davinci-edit-001")
    assert enc.name == "p50k_edit"
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    assert enc.name == "cl100k_base"
    enc = tiktoken.encoding_for_model("gpt-4")
    assert enc.name == "cl100k_base"
    enc = tiktoken.encoding_for_model("gpt-4o")
    assert enc.name == "o200k_base"
    enc = tiktoken.encoding_for_model("gpt-oss-120b")
    assert enc.name == "o200k_harmony"


def test_encoding_for_fine_tuned_model():
    # Fine-tuned models are tokenised like their base model, so the encoding must be resolved
    # from the base model name rather than from a shorter, older prefix. See issue #474.
    def encoding_name(model_name: str) -> str:
        return tiktoken.encoding_for_model(model_name).name

    assert encoding_name("ft:gpt-4.1-mini-2025-04-14:acme::abc123") == "o200k_base"
    assert encoding_name("ft:gpt-4.1-nano:acme::abc123") == "o200k_base"
    assert encoding_name("ft:gpt-4.5-preview-2025-02-27:acme::abc123") == "o200k_base"
    assert encoding_name("ft:gpt-5-mini-2025-08-07:acme::abc123") == "o200k_base"
    assert encoding_name("ft:gpt-4o-2024-08-06:acme::abc123") == "o200k_base"
    assert encoding_name("ft:o4-mini-2025-04-16:acme::abc123") == "o200k_base"
    assert encoding_name("ft:gpt-4-0613:acme::abc123") == "cl100k_base"
    assert encoding_name("ft:gpt-3.5-turbo-0125:acme::abc123") == "cl100k_base"
    assert encoding_name("ft:davinci-002:acme::abc123") == "cl100k_base"

    # Fine-tunes of unknown base models are still an error.
    with pytest.raises(KeyError):
        tiktoken.encoding_for_model("ft:unknown-model:acme::abc123")


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])
