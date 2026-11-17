import hashlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import tiktoken.load


def _unset_cache_env(monkeypatch):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)


def test_default_cache_dir_is_user_specific(monkeypatch):
    _unset_cache_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    cache_dir = tiktoken.load.default_cache_dir()
    assert cache_dir == os.path.join(os.path.expanduser("~"), ".cache", "tiktoken")
    assert not cache_dir.startswith("/tmp")

    monkeypatch.setattr(sys, "platform", "win32")
    assert tiktoken.load.default_cache_dir() == os.path.join(
        __import__("tempfile").gettempdir(), "tiktoken"
    )


def test_old_shared_temp_cache_is_ignored(monkeypatch, tmp_path):
    # simulates an attacker pre-creating a poisoned file in the old shared
    # /tmp/data-gym-cache directory; it must not be read any more
    old_cache = Path("/tmp") / "data-gym-cache"
    old_cache.mkdir(parents=True, exist_ok=True)
    url = "https://example.com/poison.tiktoken"
    key = hashlib.sha1(url.encode()).hexdigest()
    poison = b"AAAAIDE=\n"
    (old_cache / key).write_bytes(poison)

    _unset_cache_env(monkeypatch)
    new_cache = tmp_path / "new-cache"
    monkeypatch.setattr(tiktoken.load, "default_cache_dir", lambda: str(new_cache))
    good = b"AAAAIDEyCg==\n"

    def fake_read_file(blobpath):
        assert blobpath == url
        return good

    monkeypatch.setattr(tiktoken.load, "read_file", fake_read_file)

    # the poisoned shared /tmp cache must be ignored entirely
    contents = tiktoken.load.read_file_cached(url)
    assert contents == good


def test_cached_file_without_hash_warns(monkeypatch, tmp_path):
    _unset_cache_env(monkeypatch)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    url = "https://example.com/nohash.tiktoken"
    key = hashlib.sha1(url.encode()).hexdigest()
    (tmp_path / key).write_bytes(b"data")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning) as exc_info:
            tiktoken.load.read_file_cached(url, expected_hash=None)
    assert "without hash verification" in str(exc_info.value)


def test_cached_file_with_matching_hash_does_not_warn(monkeypatch, tmp_path):
    import hashlib as _hashlib

    _unset_cache_env(monkeypatch)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    url = "https://example.com/hash.tiktoken"
    data = b"data"
    key = hashlib.sha1(url.encode()).hexdigest()
    (tmp_path / key).write_bytes(data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        contents = tiktoken.load.read_file_cached(
            url, expected_hash=_hashlib.sha256(data).hexdigest()
        )
    assert contents == data


def test_tampered_cached_file_with_hash_is_refetched(monkeypatch, tmp_path):
    _unset_cache_env(monkeypatch)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    url = "https://example.com/tampered.tiktoken"
    data = b"data"
    expected_hash = hashlib.sha256(data).hexdigest()
    key = hashlib.sha1(url.encode()).hexdigest()
    (tmp_path / key).write_bytes(b"tampered")
    monkeypatch.setattr(tiktoken.load, "read_file", lambda blobpath: data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        contents = tiktoken.load.read_file_cached(url, expected_hash=expected_hash)
    assert contents == data
    assert (tmp_path / key).read_bytes() == data
