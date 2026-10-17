"""Fast query routing and per-request time budget helpers."""

import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dotenv import load_dotenv

load_dotenv()

QUERY_TIMEOUT_SECONDS = float(os.getenv("QUERY_TIMEOUT_SECONDS", "40"))
MCP_SEARCH_TIMEOUT_SECONDS = float(os.getenv("MCP_SEARCH_TIMEOUT_SECONDS", "12"))
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15"))
EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "8"))

_deadline = ContextVar("query_deadline", default=None)


class QueryDeadlineExceeded(TimeoutError):
    pass


def set_deadline(deadline):
    _deadline.set(deadline)


def remaining_seconds(limit=None):
    deadline = _deadline.get()
    remaining = deadline - time.monotonic() if deadline is not None else QUERY_TIMEOUT_SECONDS
    if remaining <= 0:
        raise QueryDeadlineExceeded("Research query exceeded its time limit. Please try again.")
    return min(remaining, limit) if limit is not None else remaining


@contextmanager
def timed_stage(name):
    started = time.monotonic()
    try:
        yield
    finally:
        from rag_system.observability import event
        event("pipeline_stage", stage=name, duration_ms=round((time.monotonic() - started) * 1000, 2))


def quick_reply(query):
    normalized = re.sub(r"[!?.,]+$", "", query.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    greetings = {"hi", "hi there", "hello", "hello there", "hey", "hey there", "good morning", "good afternoon", "good evening"}
    service_checks = {
        "are you working", "are you online", "are you there", "is this working",
        "is the service working", "is it working", "who are you", "what can you do", "how are you",
    }
    if normalized in greetings:
        return "Hello! I can help find and explain academic research papers. What topic are you exploring?"
    if normalized in service_checks:
        return "I can help search and explain academic research papers. For service status, check /api/health."
    return None


_RESEARCH_TERMS = re.compile(
    r"\b(?:paper|papers|research|study|studies|literature|arxiv|publication|citation|"
    r"scientific|academic|experiment|algorithm|model|models|neural|machine learning|"
    r"deep learning|transformer|vision|retrieval|augmentation|rag|ai|nlp|"
    r"security|cyber|biology|physics|chemistry|quantum|climate|medical|"
    r"healthcare|language processing|computing|dataset|benchmark|methodology|"
    r"autoregressive)\b",
    re.I,
)


def allows_live_research(query, *, paper_id=None):
    """Conservative gate: local retrieval still handles topics outside this vocabulary."""
    if quick_reply(query) is not None:
        return False
    if re.match(r"^\s*(?:are you|who are you|what is your|how are you|is this|is your)\b", query, re.I):
        return False
    if paper_id or re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", query):
        return True
    return bool(_RESEARCH_TERMS.search(query))
