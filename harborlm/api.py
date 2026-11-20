# language: Python 3.11, file: harborlm/api.py, target: Linux / Docker Desktop
"""HTTP client for POST /v1/chat/completions (stream SSE, else one-shot)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any, Mapping, TypeVar

import httpx

from harborlm.messages import (
    build_chat_completions,
    extract_choice_text,
    extract_delta_text,
    extract_usage,
)


class ChatError(RuntimeError):
    """Backend refused or returned an unusable body."""


CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
T = TypeVar("T")


def _validate_timeout(timeout: float, name: str = "timeout") -> float:
    value = float(timeout)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _iter_sse_lines(response: httpx.Response) -> Iterator[str]:
    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[5:].strip()
        else:
            yield line


def _looks_like_sse(response: httpx.Response, first_bytes: str) -> bool:
    ctype = response.headers.get("content-type", "").lower()
    if "text/event-stream" in ctype:
        return True
    stripped = first_bytes.lstrip()
    return stripped.startswith("data:") or stripped.startswith(":")


class LlamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 300.0,
        connect_timeout: float = 5.0,
        connect_retries: int = 1,
        retry_backoff: float = 0.15,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url or not str(base_url).strip():
            raise ValueError("base_url is required")
        if connect_retries < 0:
            raise ValueError("connect_retries must be >= 0")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be >= 0")
        self.base_url = base_url.strip().rstrip("/")
        self.read_timeout = _validate_timeout(timeout, "timeout")
        self.connect_timeout = _validate_timeout(connect_timeout, "connect_timeout")
        self.connect_retries = int(connect_retries)
        self.retry_backoff = float(retry_backoff)
        self.timeout = httpx.Timeout(self.read_timeout, connect=self.connect_timeout)
        self._own = client is None
        self._http = client or httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "harborlm/0.1.0", "Accept": "application/json"},
        )

    def _with_connect_retry(self, op: Callable[[], T]) -> T:
        last: httpx.HTTPError | None = None
        attempts = self.connect_retries + 1
        for i in range(attempts):
            try:
                return op()
            except CONNECT_ERRORS as exc:
                last = exc
                if i >= attempts - 1:
                    raise ChatError(f"connect failed: {exc}") from exc
                if self.retry_backoff > 0:
                    time.sleep(self.retry_backoff)
        raise ChatError(f"connect failed: {last}") from last

    def close(self) -> None:
        if self._own:
            self._http.close()

    def __enter__(self) -> LlamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self,
        model: str,
        messages: list[Mapping[str, Any]],
        *,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int | None = 256,
    ) -> tuple[str, dict[str, int]]:
        """Full reply + usage. Prefers stream; falls back to one-shot."""
        parts: list[str] = []
        usage: dict[str, int] = {}
        for kind, payload in self.chat_events(
            model,
            messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if kind == "delta":
                parts.append(str(payload))
            elif kind == "usage":
                usage.update(payload)  # type: ignore[arg-type]
        return "".join(parts), usage

    def chat_events(
        self,
        model: str,
        messages: list[Mapping[str, Any]],
        *,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int | None = 256,
    ) -> Iterator[tuple[str, Any]]:
        """Yield ('delta', str) and optionally ('usage', dict)."""
        url = completions_url(self.base_url)
        payload = build_chat_completions(
            model,
            messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if stream:
            try:
                yield from self._stream(url, payload)
                return
            except ChatError:
                payload = dict(payload)
                payload["stream"] = False
                yield from self._oneshot(url, payload)
                return
        yield from self._oneshot(url, payload)

    def _oneshot(self, url: str, payload: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        try:
            response = self._with_connect_retry(lambda: self._http.post(url, json=payload))
        except ChatError:
            raise
        except httpx.HTTPError as exc:
            raise ChatError(f"request failed: {exc}") from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise ChatError(self._error_text(response))
        try:
            body = response.json()
        except ValueError as exc:
            raise ChatError("completion body is not JSON") from exc
        if not isinstance(body, Mapping):
            raise ChatError("completion body is not an object")
        try:
            text = extract_choice_text(body)
        except ValueError as exc:
            raise ChatError(str(exc)) from exc
        yield ("delta", text)
        usage = extract_usage(body)
        if usage:
            yield ("usage", usage)

    def _stream(self, url: str, payload: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        attempts = self.connect_retries + 1
        last_connect: httpx.HTTPError | None = None
        for i in range(attempts):
            yielded = False
            try:
                with self._http.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code < 200 or response.status_code >= 300:
                        raw = response.read()
                        raise ChatError(
                            f"http {response.status_code}: {raw.decode('utf-8', 'replace')[:400]}"
                        )
                    assembled: list[str] = []
                    usage: dict[str, int] = {}
                    saw_data = False
                    for data in _iter_sse_lines(response):
                        if data == "[DONE]":
                            saw_data = True
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk, Mapping):
                            continue
                        saw_data = True
                        delta = extract_delta_text(chunk)
                        if delta:
                            assembled.append(delta)
                            yielded = True
                            yield ("delta", delta)
                        extra = extract_usage(chunk)
                        if extra:
                            usage.update(extra)
                        timings = chunk.get("timings")
                        if isinstance(timings, Mapping):
                            predicted = timings.get("predicted_n")
                            if isinstance(predicted, (int, float)):
                                usage["completion_tokens"] = int(predicted)
                    if usage:
                        yield ("usage", usage)
                    if not saw_data and not assembled:
                        raise ChatError("empty stream")
                return
            except ChatError:
                raise
            except CONNECT_ERRORS as exc:
                last_connect = exc
                if yielded or i >= attempts - 1:
                    raise ChatError(f"connect failed: {exc}") from exc
                if self.retry_backoff > 0:
                    time.sleep(self.retry_backoff)
            except httpx.HTTPError as exc:
                raise ChatError(f"request failed: {exc}") from exc
        raise ChatError(f"connect failed: {last_connect}") from last_connect

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, Mapping):
                err = body.get("error")
                if isinstance(err, Mapping) and "message" in err:
                    return f"http {response.status_code}: {err['message']}"
                if isinstance(err, str):
                    return f"http {response.status_code}: {err}"
        except ValueError:
            pass
        return f"http {response.status_code}: {response.text[:400]}"
