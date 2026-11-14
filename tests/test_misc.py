import subprocess
import sys

import tiktoken
from tiktoken import registry


def test_clear_registry():
    enc = tiktoken.get_encoding("gpt2")
    assert registry.ENCODINGS["gpt2"] is enc

    tiktoken.clear_registry()
    assert "gpt2" not in registry.ENCODINGS

    # Clearing is idempotent and safe to call again.
    tiktoken.clear_registry()

    # The encoding is re-created lazily and still works.
    new_enc = tiktoken.get_encoding("gpt2")
    assert new_enc.encode("hello world") == [31373, 995]
    assert registry.ENCODINGS["gpt2"] is new_enc


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


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])
