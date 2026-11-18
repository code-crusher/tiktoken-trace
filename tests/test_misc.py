import subprocess
import sys

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


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])


def test_optional_regex_dependency():
    # `regex` does not have prebuilt wheels on all platforms tiktoken supports
    # (e.g. Windows on ARM64, see #403), so it must remain an optional dependency.
    prog = """
import importlib.metadata
import sys

# Simulate the package being unavailable on this platform
class _BlockRegex:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "regex" or fullname.startswith("regex."):
            raise ImportError("regex is blocked")
        return None

sys.meta_path.insert(0, _BlockRegex())
sys.modules.pop("regex", None)

# The installed distribution must not require `regex`
for requirement in importlib.metadata.requires("tiktoken") or []:
    if requirement.partition(";")[0].strip().lower().startswith("regex"):
        assert "extra" in requirement, requirement

import tiktoken

enc = tiktoken.get_encoding("gpt2")
assert enc.encode("hello world") == [31373, 995]
assert enc.decode([31373, 995]) == "hello world"
text = "hello " + enc.decode([50256])
assert enc.encode(text, allowed_special="all") == [31373, 220, 50256]
assert enc.encode_ordinary(text) != [31373, 220, 50256]

import re
assert isinstance(tiktoken.core._special_token_regex(frozenset(enc.special_tokens_set)), re.Pattern)

try:
    enc.encode(text)
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError for disallowed special token")

try:
    import tiktoken._educational
except ImportError as e:
    assert "tiktoken[regex]" in str(e)
else:
    raise AssertionError("expected ImportError from tiktoken._educational")

assert "regex" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])
