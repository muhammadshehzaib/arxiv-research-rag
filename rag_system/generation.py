"""Bounded Gemini generation with explicit model fallback and usage accounting."""
import os
import time
import google.generativeai as genai

from rag_system.observability import event, record_generation
from rag_system.query_policy import remaining_seconds, GEMINI_TIMEOUT_SECONDS


def retryable(exc):
    return isinstance(exc, TimeoutError) or any(word in (type(exc).__name__ + " " + str(exc)).lower()
        for word in ("429", "503", "quota", "resourceexhausted", "unavailable", "deadlineexceeded", "timeout"))


def generate_content(prompt, model_name=None, **kwargs):
    primary = model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")
    models = list(dict.fromkeys([primary] + [name.strip() for name in
        os.getenv("LLM_FALLBACK_MODELS", "").split(",") if name.strip()]))
    last_error = None
    provided_options = kwargs.pop("request_options", {})
    primary_model = kwargs.pop("primary_model", None)
    for index, name in enumerate(models):
        for attempt in range(2):
            options = dict(provided_options)
            options["timeout"] = min(float(options.get("timeout", GEMINI_TIMEOUT_SECONDS)),
                                     remaining_seconds(GEMINI_TIMEOUT_SECONDS))
            try:
                model = primary_model if index == 0 and primary_model is not None else genai.GenerativeModel(model_name=name)
                response = model.generate_content(
                    prompt, request_options=options, **kwargs)
                record_generation(name, response=response)
                if index:
                    event("generation_fallback_succeeded", model=name)
                return response
            except Exception as exc:
                record_generation(name, error=exc)
                last_error = exc
                if not retryable(exc):
                    raise
                if attempt == 0:
                    time.sleep(min(0.25, remaining_seconds(0.25)))
        if index + 1 < len(models):
            event("generation_fallback", from_model=name, to_model=models[index + 1])
    raise last_error
