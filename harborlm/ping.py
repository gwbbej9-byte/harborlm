# language: Python 3.11, file: harborlm/ping.py, target: Linux / Docker Desktop
"""GET /v1/models or /health against an OpenAI-compatible llama.cpp server."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse

import httpx

CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


def _strip_slash(url: str) -> str:
    return url.strip().rstrip("/")


def root_from_v1(url: str) -> str:
    base = _strip_slash(url)
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base


def ping_targets(url: str) -> list[str]:
    """Ordered URLs to try. Models first, then health."""
    base = _strip_slash(url)
    root = root_from_v1(base)
    v1 = base if base.endswith("/v1") else f"{root}/v1"
    seen: list[str] = []
    for item in (
        f"{v1}/models",
        f"{root}/health",
        f"{v1}/health",
        f"{root}/v1/models",
    ):
        if item not in seen:
            seen.append(item)
    return seen


def parse_models_payload(payload: Mapping[str, Any]) -> list[str]:
    """OpenAI list object → model ids. Empty list if the shape is unknown."""
    ids: list[str] = []
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, Mapping):
                continue
            ident = item.get("id") or item.get("name")
            if isinstance(ident, str) and ident:
                ids.append(ident)
        return ids
    models = payload.get("models")
    if isinstance(models, list):
        for item in models:
            if isinstance(item, str) and item:
                ids.append(item)
            elif isinstance(item, Mapping):
                ident = item.get("id") or item.get("name")
                if isinstance(ident, str) and ident:
                    ids.append(ident)
    return ids


@dataclass
class HealthParse:
    ok: bool
    status: str
    detail: str = ""


def parse_health_payload(payload: Mapping[str, Any] | str | None, http_status: int = 200) -> HealthParse:
    if http_status >= 200 and http_status < 300:
        if payload is None or payload == "":
            return HealthParse(True, "ok", "empty body")
        if isinstance(payload, str):
            text = payload.strip().lower()
            if text in {"ok", "ok\n", "healthy"} or "ok" in text[:32]:
                return HealthParse(True, "ok", payload.strip()[:200])
            return HealthParse(True, "ok", payload.strip()[:200])
        if isinstance(payload, Mapping):
            status = payload.get("status") or payload.get("state") or "ok"
            status_s = str(status)
            ok = status_s.lower() in {"ok", "healthy", "ready", "up"} or http_status < 300
            return HealthParse(ok, status_s, str(payload)[:200])
    return HealthParse(False, "down", f"http {http_status}")


@dataclass
class PingResult:
    ok: bool
    url: str
    http_status: int
    models: list[str] = field(default_factory=list)
    health: HealthParse | None = None
    error: str = ""
    tried: list[str] = field(default_factory=list)

    def format_lines(self) -> str:
        if self.ok:
            bits = [f"ok  {self.url}  {self.http_status}"]
            if self.models:
                bits.append("models: " + ", ".join(self.models))
            elif self.health:
                bits.append(f"health: {self.health.status}")
            return "\n".join(bits)
        lines = [f"fail  {self.error or 'unreachable'}"]
        if self.tried:
            lines.append("tried: " + " ".join(self.tried))
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "url": self.url,
            "http_status": self.http_status,
            "models": list(self.models),
            "health": self.health.status if self.health else None,
            "error": self.error,
            "tried": list(self.tried),
        }

    def format_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


def _decode_body(response: httpx.Response) -> Any:
    ctype = response.headers.get("content-type", "")
    text = response.text
    if "json" in ctype or (text[:1] in "{["):
        try:
            return response.json()
        except ValueError:
            return text
    return text


def _get_with_retry(
    http: httpx.Client,
    target: str,
    *,
    connect_retries: int = 1,
    retry_backoff: float = 0.15,
) -> httpx.Response:
    attempts = connect_retries + 1
    last: httpx.HTTPError | None = None
    for i in range(attempts):
        try:
            return http.get(target)
        except CONNECT_ERRORS as exc:
            last = exc
            if i >= attempts - 1:
                raise
            if retry_backoff > 0:
                time.sleep(retry_backoff)
    assert last is not None
    raise last


def ping(
    url: str,
    *,
    timeout: float = 3.0,
    client: httpx.Client | None = None,
    connect_retries: int = 1,
    retry_backoff: float = 0.15,
) -> PingResult:
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    if connect_retries < 0:
        raise ValueError("connect_retries must be >= 0")
    targets = ping_targets(url)
    own = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    last_error = "no targets"
    last_status = 0
    last_url = targets[0] if targets else url
    try:
        for target in targets:
            try:
                response = _get_with_retry(
                    http,
                    target,
                    connect_retries=connect_retries,
                    retry_backoff=retry_backoff,
                )
            except httpx.HTTPError as exc:
                last_error = str(exc)
                last_url = target
                continue
            last_status = response.status_code
            last_url = target
            if response.status_code >= 500:
                last_error = f"http {response.status_code}"
                continue
            body = _decode_body(response)
            parsed = urlparse(target)
            path = parsed.path.rstrip("/")
            if path.endswith("/models"):
                if response.status_code >= 400:
                    last_error = f"http {response.status_code}"
                    continue
                models = parse_models_payload(body) if isinstance(body, Mapping) else []
                return PingResult(
                    ok=True,
                    url=target,
                    http_status=response.status_code,
                    models=models,
                    tried=targets,
                )
            health = parse_health_payload(
                body if isinstance(body, (Mapping, str)) else str(body),
                response.status_code,
            )
            if health.ok:
                return PingResult(
                    ok=True,
                    url=target,
                    http_status=response.status_code,
                    health=health,
                    tried=targets,
                )
            last_error = health.detail or health.status
        return PingResult(
            ok=False,
            url=last_url,
            http_status=last_status,
            error=last_error,
            tried=targets,
        )
    finally:
        if own:
            http.close()


def describe_backend(url: str) -> str:
    parsed = urlparse(_strip_slash(url))
    if parsed.scheme in {"demo"}:
        return "demo"
    host = parsed.netloc or parsed.path or url
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", "")) or host
