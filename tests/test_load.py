import hashlib
import os

import pytest

from tiktoken.load import OfflineCacheMissError, read_file, read_file_cached

URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
CONTENTS = b"SGVsbG8= 0\n"


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    # Isolate the tests from the default cache and any cache the user configured.
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    monkeypatch.delenv("TIKTOKEN_OFFLINE", raising=False)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr("requests.get", fail)


def test_cache_hit_by_sha1(cache_dir, no_network):
    cache_path = cache_dir / hashlib.sha1(URL.encode()).hexdigest()
    cache_path.write_bytes(CONTENTS)

    assert read_file_cached(URL) == CONTENTS


def test_cache_hit_by_basename(cache_dir, no_network):
    # Users commonly pre-populate a cache with the original file names.
    (cache_dir / os.path.basename(URL)).write_bytes(CONTENTS)

    assert read_file_cached(URL) == CONTENTS


def test_cache_hit_by_basename_with_expected_hash(cache_dir, no_network):
    (cache_dir / os.path.basename(URL)).write_bytes(CONTENTS)

    assert read_file_cached(URL, hashlib.sha256(CONTENTS).hexdigest()) == CONTENTS


def test_sha1_cache_preferred_over_basename(cache_dir, no_network):
    (cache_dir / hashlib.sha1(URL.encode()).hexdigest()).write_bytes(CONTENTS)
    (cache_dir / os.path.basename(URL)).write_bytes(b"stale")

    assert read_file_cached(URL) == CONTENTS


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_offline_cache_hit_does_not_use_network(cache_dir, no_network, monkeypatch, value):
    monkeypatch.setenv("TIKTOKEN_OFFLINE", value)
    (cache_dir / os.path.basename(URL)).write_bytes(CONTENTS)

    assert read_file_cached(URL) == CONTENTS


def test_offline_cache_miss_does_not_use_network(cache_dir, no_network, monkeypatch):
    monkeypatch.setenv("TIKTOKEN_OFFLINE", "1")

    with pytest.raises(OfflineCacheMissError, match="TIKTOKEN_OFFLINE"):
        read_file_cached(URL)


def test_offline_cache_miss_is_file_not_found_and_value_error(cache_dir, no_network, monkeypatch):
    monkeypatch.setenv("TIKTOKEN_OFFLINE", "1")

    with pytest.raises(OfflineCacheMissError) as excinfo:
        read_file_cached(URL)

    assert isinstance(excinfo.value, FileNotFoundError)
    assert isinstance(excinfo.value, ValueError)


def test_offline_with_caching_disabled_does_not_use_network(no_network, monkeypatch):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    monkeypatch.setenv("TIKTOKEN_OFFLINE", "1")

    with pytest.raises(OfflineCacheMissError, match="TIKTOKEN_OFFLINE"):
        read_file_cached(URL)


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_offline_disabled_values(cache_dir, monkeypatch, value):
    monkeypatch.setenv("TIKTOKEN_OFFLINE", value)
    (cache_dir / os.path.basename(URL)).write_bytes(CONTENTS)

    assert read_file_cached(URL) == CONTENTS


def test_offline_mode_still_reads_local_files(tmp_path, monkeypatch):
    monkeypatch.setenv("TIKTOKEN_OFFLINE", "1")
    local_file = tmp_path / "vocab.tiktoken"
    local_file.write_bytes(CONTENTS)

    assert read_file(str(local_file)) == CONTENTS
