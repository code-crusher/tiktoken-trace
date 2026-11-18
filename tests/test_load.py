import socket
import threading

import pytest

import tiktoken.load


@pytest.fixture
def stalled_server():
    """A server that accepts connections but never sends a response."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    conns = []

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            conns.append(conn)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.getsockname()[1]}/vocab"
    finally:
        srv.close()
        for conn in conns:
            conn.close()


class _FakeResponse:
    content = b"contents"

    def raise_for_status(self):
        pass


def test_read_file_http_timeout_default(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse()

    monkeypatch.delenv("TIKTOKEN_HTTP_TIMEOUT", raising=False)
    monkeypatch.setattr("requests.get", fake_get)

    assert tiktoken.load.read_file("https://example.com/vocab") == b"contents"
    assert calls == [
        ("https://example.com/vocab", {"timeout": tiktoken.load.DEFAULT_HTTP_TIMEOUT})
    ]
    assert tiktoken.load.DEFAULT_HTTP_TIMEOUT > 0


def test_read_file_http_timeout_env_override(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse()

    monkeypatch.setenv("TIKTOKEN_HTTP_TIMEOUT", "5.5")
    monkeypatch.setattr("requests.get", fake_get)

    assert tiktoken.load.read_file("http://example.com/vocab") == b"contents"
    assert calls == [("http://example.com/vocab", {"timeout": 5.5})]


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1", "nan", "inf", ""])
def test_read_file_http_timeout_invalid_env_value(monkeypatch, value):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse()

    monkeypatch.setenv("TIKTOKEN_HTTP_TIMEOUT", value)
    monkeypatch.setattr("requests.get", fake_get)

    assert tiktoken.load.read_file("https://example.com/vocab") == b"contents"
    assert calls == [
        ("https://example.com/vocab", {"timeout": tiktoken.load.DEFAULT_HTTP_TIMEOUT})
    ]


def test_read_file_local_file_has_no_timeout(monkeypatch, tmp_path):
    path = tmp_path / "vocab"
    path.write_bytes(b"local")

    def fake_get(url, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("requests.get should not be called for local files")

    monkeypatch.setattr("requests.get", fake_get)
    assert tiktoken.load.read_file(str(path)) == b"local"


def test_read_file_http_times_out_on_stalled_server(monkeypatch, stalled_server):
    # A stalled connection must raise instead of blocking forever; see issue #413.
    monkeypatch.setenv("TIKTOKEN_HTTP_TIMEOUT", "0.5")

    with pytest.raises(Exception) as exc_info:
        tiktoken.load.read_file(stalled_server)
    assert type(exc_info.value).__module__.startswith("requests")
    assert "timed out" in str(exc_info.value).lower()
