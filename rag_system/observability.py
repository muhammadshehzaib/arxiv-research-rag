"""Request-linked events and bounded per-process operational metrics."""
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import Counter, deque
from contextvars import ContextVar

request_id = ContextVar("request_id", default=None)
logger = logging.getLogger("rag.events")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
_lock = threading.Lock()
_counts = Counter()
_models = {}
_failures = deque(maxlen=1000)
_latencies = deque(maxlen=1000)


def event(name, **fields):
    logger.info(json.dumps({"timestamp": time.time(), "request_id": request_id.get(),
                            "event": name, **fields}, allow_nan=False))


def record_failure(stage, error_type):
    with _lock:
        _counts["errors_total"] += 1
        _failures.append(time.time())
    event("failure", stage=stage, error_type=error_type)


def record_generation(model, response=None, error=None):
    usage = getattr(response, "usage_metadata", None)
    def tokens(field):
        value = getattr(usage, field, None)
        return int(value) if isinstance(value, int) and value >= 0 else 0
    input_tokens, output_tokens = tokens("prompt_token_count"), tokens("candidates_token_count")
    total_tokens = tokens("total_token_count")
    cost = None
    try:
        pricing = json.loads(os.getenv("RAG_MODEL_PRICING_JSON", "{}")).get(model)
        if pricing and usage is not None and total_tokens > 0:
            input_price, output_price = float(pricing["input_per_million"]), float(pricing["output_per_million"])
            if all(math.isfinite(v) and v >= 0 for v in (input_price, output_price)):
                cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    except (ValueError, TypeError, KeyError):
        pass
    with _lock:
        _counts["generation_calls_total"] += 1
        stats = _models.setdefault(model, dict(calls=0, failures=0, input_tokens=0, output_tokens=0,
                                              total_tokens=0, estimated_cost_usd=0.0, unpriced_calls=0))
        stats["calls"] += 1
        stats["failures"] += int(error is not None)
        stats["input_tokens"] += input_tokens
        stats["output_tokens"] += output_tokens
        stats["total_tokens"] += total_tokens
        stats["unpriced_calls"] += int(cost is None)
        stats["estimated_cost_usd"] += cost or 0.0
    event("generation", model=model, input_tokens=input_tokens, output_tokens=output_tokens,
          total_tokens=total_tokens, estimated_cost_usd=cost, error_type=type(error).__name__ if error else None)
    if error is not None:
        record_failure("generation", type(error).__name__)


def metrics_snapshot():
    with _lock:
        recent = sum(t >= time.time() - 60 for t in _failures)
        threshold = int(os.getenv("RAG_ALERT_FAILURE_THRESHOLD", "5"))
        alerts = ([{"name": "elevated_errors", "errors_last_minute": recent}]
                  if recent >= threshold else [])
        models = json.loads(json.dumps(_models))
        for values in models.values():
            if values["unpriced_calls"]:
                values["priced_calls_cost_usd"] = values["estimated_cost_usd"]
                values["estimated_cost_usd"] = None
        result = {"scope": "server_process", "counts": dict(_counts), "generation_models": models,
                  "recent_request_latency_ms": list(_latencies), "active_alerts": alerts}
    if alerts:
        event("alert", alerts=alerts)
    return result


class ObservabilityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        rid = supplied if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", supplied) else uuid.uuid4().hex
        token = request_id.set(rid)
        started, status = time.monotonic(), 500
        async def traced_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [(b"x-request-id", rid.encode())]
            await send(message)
        try:
            await self.app(scope, receive, traced_send)
        except Exception as exc:
            record_failure("http", type(exc).__name__)
            raise
        finally:
            elapsed = round((time.monotonic() - started) * 1000, 2)
            path = scope.get("path", "")
            path = path if path in {"/api/query", "/api/query/stream", "/api/eval/run", "/api/health", "/api/metrics"} else "/other"
            with _lock:
                _counts["http_requests_total"] += 1
                _counts[f"http_status_{status}"] += 1
                _latencies.append(elapsed)
            if status >= 500:
                record_failure("http_response", str(status))
            event("request_complete", path=path, status=status, duration_ms=elapsed)
            request_id.reset(token)
