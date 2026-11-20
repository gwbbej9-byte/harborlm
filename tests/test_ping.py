# language: Python 3.11, file: tests/test_ping.py, target: pytest
from __future__ import annotations

import json

import httpx
import pytest

from harborlm.cli import main
from harborlm.ping import (
    parse_health_payload,
    parse_models_payload,
    ping,
    ping_targets,
)


def test_ping_targets_from_v1() -> None:
    targets = ping_targets("http://127.0.0.1:8080/v1/")
    assert targets[0] == "http://127.0.0.1:8080/v1/models"
    assert "http://127.0.0.1:8080/health" in targets


def test_ping_targets_from_root() -> None:
    targets = ping_targets("http://127.0.0.1:8080")
    assert "http://127.0.0.1:8080/v1/models" in targets
    assert "http://127.0.0.1:8080/health" in targets


def test_parse_openai_models_list() -> None:
    ids = parse_models_payload(
        {
            "object": "list",
            "data": [
                {"id": "tinyllama", "object": "model"},
                {"name": "alias-only"},
            ],
        }
    )
    assert ids == ["tinyllama", "alias-only"]


def test_parse_models_llama_cpp_models_key() -> None:
    ids = parse_models_payload({"models": ["tinyllama", {"id": "other"}]})
    assert ids == ["tinyllama", "other"]


def test_parse_health_json_and_plain() -> None:
    assert parse_health_payload({"status": "ok"}).ok
    assert parse_health_payload("ok", 200).ok
    assert parse_health_payload(None, 200).ok
    assert not parse_health_payload("nope", 503).ok


def test_ping_models_httpx_mock() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "tinyllama"}]},
            )
        return httpx.Response(404, text="no")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = ping("http://127.0.0.1:8080/v1", client=client)
    assert result.ok
    assert result.models == ["tinyllama"]
    assert result.http_status == 200
    assert "/v1/models" in result.url


def test_ping_falls_back_to_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(404, json={"error": "missing"})
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = ping("http://127.0.0.1:8080/v1", client=client)
    assert result.ok
    assert result.health is not None and result.health.ok
    assert result.models == []


def test_ping_retries_connect_then_ok() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"object": "list", "data": [{"id": "tinyllama"}]})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = ping(
            "http://127.0.0.1:8080/v1",
            client=client,
            retry_backoff=0,
        )
    assert result.ok
    assert hits["n"] == 2


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    real_client = httpx.Client

    def factory(**kwargs: object) -> httpx.Client:
        kwargs.pop("transport", None)
        timeout = kwargs.get("timeout", 3.0)
        return real_client(transport=transport, timeout=timeout, follow_redirects=True)

    monkeypatch.setattr("harborlm.ping.httpx.Client", factory)


def test_ping_cli_exit_0_mocked_http(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": [{"id": "tinyllama"}]})

    _patch_client(monkeypatch, handler)
    code = main(["ping", "--url", "http://127.0.0.1:8080/v1", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["models"] == ["tinyllama"]


def test_ping_cli_exit_1_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _patch_client(monkeypatch, handler)
    code = main(["ping", "--url", "http://127.0.0.1:9/v1", "--timeout", "0.2"])
    assert code == 1
