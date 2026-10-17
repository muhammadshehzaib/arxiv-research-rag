"""Local-only defaults and explicit bearer authentication for hosted API access."""
import hmac
import ipaddress
import os
import threading
import time
from collections import OrderedDict, deque

from starlette.responses import JSONResponse

_buckets = OrderedDict()
_lock = threading.Lock()


def rate_allowed(client, group, limit, window=60, now=None):
    now = time.monotonic() if now is None else now
    with _lock:
        key = (client, group)
        bucket = _buckets.pop(key, deque())
        while bucket and bucket[0] <= now - window:
            bucket.popleft()
        allowed = len(bucket) < limit
        if allowed:
            bucket.append(now)
        _buckets[key] = bucket
        while len(_buckets) > 10000:
            _buckets.popitem(last=False)
        return allowed


async def guard_api(request, call_next):
    if not request.url.path.startswith("/api/") or request.url.path == "/api/health":
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)
    client = request.client.host if request.client else "unknown"
    limit = int(os.getenv("RAG_RATE_LIMIT_PER_MINUTE", "60"))
    if not rate_allowed(client, "api", limit):
        return JSONResponse({"detail": "Request limit reached. Try again later."}, 429, headers={"Retry-After": "60"})
    token = os.getenv("RAG_API_TOKEN", "")
    if token:
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            return JSONResponse({"detail": "Authentication required."}, 401,
                                headers={"WWW-Authenticate": "Bearer"})
    else:
        host = request.url.hostname
        try:
            local_client = ipaddress.ip_address(client).is_private
        except ValueError:
            local_client = client == "testclient"
        if (os.getenv("RAG_AUTH_MODE", "local") != "local" or not local_client
                or host not in {"localhost", "127.0.0.1", "::1", "testserver"}):
            return JSONResponse({"detail": "Hosted API access requires RAG_API_TOKEN."}, 503)
    if request.url.path == "/api/eval/run" and not rate_allowed(client, "evaluation", 2, window=3600):
        return JSONResponse({"detail": "Evaluation limit reached."}, 429, headers={"Retry-After": "3600"})
    if request.method in {"POST", "PUT", "PATCH"}:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                return JSONResponse({"detail": "Request body exceeds 64 KiB."}, 413)
        request._body = bytes(body)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response
