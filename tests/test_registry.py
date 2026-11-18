import threading
import time

import pytest

import tiktoken
import tiktoken.registry


@pytest.fixture
def dummy_constructors(monkeypatch):
    """Replaces the plugin constructors with ones that don't touch the network."""

    def dummy_encoding(name):
        return {
            "name": name,
            "pat_str": r"\s+|\S+",
            "mergeable_ranks": {b"a": 0, b"b": 1},
            "special_tokens": {},
        }

    monkeypatch.setattr(
        tiktoken.registry,
        "ENCODING_CONSTRUCTORS",
        {
            "dummy_a": lambda: dummy_encoding("dummy_a"),
            "dummy_b": lambda: dummy_encoding("dummy_b"),
        },
    )
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})
    return dummy_encoding


def test_get_encoding_slow_constructor_does_not_block_other_encodings(
    dummy_constructors, monkeypatch
):
    # A constructor that stalls (e.g. a slow download) must not block lookups of
    # unrelated encodings; see issue #413.
    entered = threading.Event()
    release = threading.Event()

    def slow_constructor():
        entered.set()
        release.wait(timeout=10)
        return dummy_constructors("dummy_a")

    constructors = dict(tiktoken.registry.ENCODING_CONSTRUCTORS)
    constructors["dummy_a"] = slow_constructor
    monkeypatch.setattr(tiktoken.registry, "ENCODING_CONSTRUCTORS", constructors)

    results = {}

    def get_slow():
        results["slow"] = tiktoken.get_encoding("dummy_a")

    slow_thread = threading.Thread(target=get_slow, daemon=True)
    slow_thread.start()
    assert entered.wait(timeout=5)

    other_thread = threading.Thread(
        target=lambda: results.setdefault("other", tiktoken.get_encoding("dummy_b")),
        daemon=True,
    )
    start = time.monotonic()
    other_thread.start()
    other_thread.join(timeout=5)
    elapsed = time.monotonic() - start

    assert not other_thread.is_alive(), "get_encoding blocked on an unrelated encoding"
    assert elapsed < 5
    assert results["other"].name == "dummy_b"

    # Unblock the slow constructor and make sure it still gets registered.
    release.set()
    slow_thread.join(timeout=10)
    assert not slow_thread.is_alive()
    assert results["slow"].name == "dummy_a"
    assert tiktoken.registry.ENCODINGS["dummy_a"] is results["slow"]
    assert tiktoken.get_encoding("dummy_a") is results["slow"]


def test_get_encoding_constructs_each_encoding_once(dummy_constructors, monkeypatch):
    calls = []
    lock = threading.Lock()

    def counting_constructor(name):
        def constructor():
            with lock:
                calls.append(name)
            time.sleep(0.05)
            return dummy_constructors(name)

        return constructor

    monkeypatch.setattr(
        tiktoken.registry,
        "ENCODING_CONSTRUCTORS",
        {
            "dummy_a": counting_constructor("dummy_a"),
            "dummy_b": counting_constructor("dummy_b"),
        },
    )

    results = []
    barrier = threading.Barrier(8)

    def get(name):
        barrier.wait(timeout=5)
        results.append(tiktoken.get_encoding(name))

    threads = [
        threading.Thread(target=get, args=("dummy_a" if i % 2 else "dummy_b",), daemon=True)
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert sorted(calls) == ["dummy_a", "dummy_b"]
    assert len(results) == 8
    for name in ("dummy_a", "dummy_b"):
        matching = [enc for enc in results if enc.name == name]
        assert len(matching) == 4
        assert all(enc is matching[0] for enc in matching)
