import os
import sys
import re
import json
import numpy as np

_corpus_size_cache = {"size": None, "mtime": None}

def setup_windows_encoding():
    """
    Configures stdout and stderr to UTF-8 encoding on Windows systems
    to prevent console encoding crashes on Unicode characters and emojis.
    """
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

def date_to_int(date_str):
    """
    Converts ISO date strings (e.g. '2024-03-15', '2024-03', '2024') to an integer YYYYMMDD.
    Returns 0 if date_str is empty or cannot be parsed.
    """
    if not date_str:
        return 0
    digits = "".join(c for c in str(date_str) if c.isdigit())
    if len(digits) >= 8:
        return int(digits[:8])
    elif len(digits) == 6:
        return int(digits + "01")
    elif len(digits) == 4:
        return int(digits + "0101")
    return 0

def clean_text(text):
    """
    Cleans and normalizes raw text:
    - Strips unpaired surrogate characters that crash Rust-based tokenizers
    - Recombines hyphenated linebreaks (e.g. 'multi-\nagent' -> 'multiagent')
    - Collapses newlines and multiple whitespace into a single space
    """
    if not text:
        return ""
    # Remove surrogate characters (U+D800 to U+DFFF)
    text = "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))
    # Remove hyphenated linebreaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Replace newlines with spaces
    text = re.sub(r'\r?\n|\r', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_text(text):
    """
    Standard alphanumeric lowercasing tokenizer for BM25 and lexical search.
    """
    if not text:
        return []
    return re.findall(r'[a-z0-9]+', text.lower())

def load_json(filepath, default=None):
    """
    Safely loads a JSON file with UTF-8 encoding.
    Returns default if missing or corrupted.
    """
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error reading JSON from {filepath}: {e}")
        return default

def save_json(filepath, data, indent=2):
    """
    Safely writes data to a JSON file with UTF-8 encoding,
    automatically creating parent directories if needed.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def to_jsonable(obj):
    """
    Recursively convert numpy scalars/arrays to native Python types
    so FastAPI's jsonable_encoder can serialize them.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_corpus_size():
    """
    Returns the current number of papers in the corpus.
    Caches based on the metadata file's mtime, so it re-reads only when the
    file actually changes on disk (not on every call).
    """
    metadata_path = os.path.join("data", "papers_metadata.json")
    if not os.path.exists(metadata_path):
        return 0
    try:
        mtime = os.path.getmtime(metadata_path)
        # If the file hasn't changed since last read, reuse cached value
        if _corpus_size_cache["mtime"] == mtime and _corpus_size_cache["size"] is not None:
            return _corpus_size_cache["size"]
        # File changed (or first call) — read it fresh
        with open(metadata_path, "r", encoding="utf-8") as f:
            size = len(json.load(f))
        _corpus_size_cache["mtime"] = mtime
        _corpus_size_cache["size"] = size
        return size
    except Exception as e:
        print(f"⚠️ Could not read corpus size: {e}")
        return 0