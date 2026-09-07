"""
calibrate_threshold.py
======================
Finds the best REFUSAL_CONFIDENCE_THRESHOLD for YOUR specific dataset by:

  1. Running a set of IN-SCOPE questions (your papers should answer these)
  2. Running a set of OUT-OF-SCOPE questions (your papers cannot answer these)
  3. For each question, capturing the raw reranker score BEFORE any threshold decision
  4. Sweeping thresholds from 0.40 to 0.95 and measuring accuracy at each step
  5. Picking the threshold where in-scope questions PASS and out-of-scope questions FAIL

Usage:
    python calibrate_threshold.py

Output:
    - A table showing accuracy at each threshold
    - The recommended threshold printed to console
    - Optionally auto-updates REFUSAL_THRESHOLD in your .env file
"""

import os
import sys
import json
import numpy as np
from dotenv import load_dotenv

# Windows UTF-8 fix
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

# Import the RAG pipeline (needed to get real reranker scores)
from query_rag import init_services, get_bm25_index, get_reranker, get_query_embedding, EMBEDDING_PROVIDER
from utils import tokenize_text

# =============================================================================
#  TEST QUESTIONS
#  Edit these lists to match YOUR corpus.
#  IN_SCOPE  -> questions your papers definitely cover (should pass threshold)
#  OUT_SCOPE -> questions completely outside your corpus (should be refused)
# =============================================================================

IN_SCOPE_QUESTIONS = [
    # RAG / Retrieval papers
    "What is Retrieval-Augmented Generation and why is it useful?",
    "How does a cross-encoder reranker improve retrieval quality?",
    "What are the main components of a RAG pipeline?",
    "Explain Reciprocal Rank Fusion for hybrid search.",
    "What is Graph RAG and how does it use citation networks?",

    # ML / NLP general (likely well covered in an arXiv AI corpus)
    "What are the advantages of transformer-based language models?",
    "How does BM25 keyword search work?",
    "What is the difference between dense and sparse retrieval?",
    "Explain the role of embeddings in semantic search.",
    "What is hallucination in large language models?",
]

OUT_SCOPE_QUESTIONS = [
    # Sport / Entertainment
    "Who won the FIFA World Cup in 2022?",
    "What are the best Netflix shows of 2024?",

    # Cooking / Lifestyle
    "How do I bake a chocolate cake?",
    "What is the healthiest breakfast for weight loss?",

    # Geography / Travel
    "What is the capital city of Australia?",
    "How many countries are in the European Union?",

    # Finance / Economy
    "What is the current price of Bitcoin?",
    "How do I file my taxes in the United States?",

    # Medical / Biology (not AI papers)
    "What is the recommended daily intake of vitamin D?",
    "How does the human immune system fight viruses?",
]

# Sigmoid helper (same formula used in query_rag.py)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# =============================================================================
#  STEP 1: Get the top reranker score for a query WITHOUT applying any threshold
# =============================================================================

def get_top_reranker_score(collection, query_text, bm25, bm25_chunks, reranker):
    """
    Runs the full retrieval + reranking pipeline for a single query and returns
    the top raw reranker score (before sigmoid conversion).
    Returns None if retrieval finds nothing.
    """
    fetch_results = 30

    # Dense retrieval
    dense_candidates = []
    try:
        if EMBEDDING_PROVIDER == "local":
            results = collection.query(query_texts=[query_text], n_results=fetch_results)
        else:
            query_vector = get_query_embedding(query_text)
            results = collection.query(query_embeddings=[query_vector], n_results=fetch_results)

        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist, cid in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                dense_candidates.append({"chunk_id": cid, "text": doc, "metadata": meta, "distance": dist})
    except Exception as e:
        print(f"      Warning: Dense search error: {e}")

    # Sparse retrieval (BM25)
    sparse_candidates = []
    if bm25 and bm25_chunks:
        try:
            tokenized_query = tokenize_text(query_text)
            scores = bm25.get_scores(tokenized_query)
            scored = [(chunk, score) for chunk, score in zip(bm25_chunks, scores) if score > 0.0]
            scored.sort(key=lambda x: x[1], reverse=True)
            sparse_candidates = scored[:fetch_results]
        except Exception as e:
            print(f"      Warning: Sparse search error: {e}")

    if not dense_candidates and not sparse_candidates:
        return None  # Truly nothing found

    # RRF blending
    dense_ranks  = {c["chunk_id"]: i + 1 for i, c in enumerate(dense_candidates)}
    sparse_ranks = {c[0]["chunk_id"]: i + 1 for i, c in enumerate(sparse_candidates)}
    all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    chunk_lookup = {}
    for c in dense_candidates:
        chunk_lookup[c["chunk_id"]] = {"text": c["text"], "metadata": c["metadata"]}
    for chunk, score in sparse_candidates:
        cid = chunk["chunk_id"]
        if cid not in chunk_lookup:
            chunk_lookup[cid] = {"text": chunk["text"], "metadata": {}}

    k = 60
    rrf_scores = []
    for cid in all_ids:
        dr = dense_ranks.get(cid)
        sr = sparse_ranks.get(cid)
        score = (1.0 / (k + dr) if dr else 0.0) + (1.0 / (k + sr) if sr else 0.0)
        rrf_scores.append((cid, score))
    rrf_scores.sort(key=lambda x: x[1], reverse=True)

    # Cross-encoder reranking
    top_30 = rrf_scores[:30]
    passages = []
    for cid, rrf_score in top_30:
        item = chunk_lookup[cid]
        passages.append({"id": cid, "text": item["text"], "metadata": item["metadata"], "rrf_score": rrf_score})

    reranked = reranker.rerank(query_text, passages)

    if not reranked:
        return None

    # Return the highest raw reranker score
    top_raw_score = reranked[0].get("score", None)
    return top_raw_score


# =============================================================================
#  STEP 2: Collect scores for all questions
# =============================================================================

def collect_scores(collection, bm25, bm25_chunks, reranker):
    in_scope_scores  = []
    out_scope_scores = []

    print("\n" + "="*70)
    print("  Collecting scores for IN-SCOPE questions...")
    print("="*70)
    for i, q in enumerate(IN_SCOPE_QUESTIONS, 1):
        print(f"  [{i:02d}/{len(IN_SCOPE_QUESTIONS)}] {q[:65]}")
        raw = get_top_reranker_score(collection, q, bm25, bm25_chunks, reranker)
        if raw is None:
            print(f"         -> No results found (treating as raw score = -10)")
            raw = -10.0
        conf = sigmoid(raw)
        print(f"         -> Raw score: {raw:+.4f}  |  Confidence: {conf*100:.1f}%")
        in_scope_scores.append({"question": q, "raw": raw, "confidence": conf})

    print("\n" + "="*70)
    print("  Collecting scores for OUT-OF-SCOPE questions...")
    print("="*70)
    for i, q in enumerate(OUT_SCOPE_QUESTIONS, 1):
        print(f"  [{i:02d}/{len(OUT_SCOPE_QUESTIONS)}] {q[:65]}")
        raw = get_top_reranker_score(collection, q, bm25, bm25_chunks, reranker)
        if raw is None:
            print(f"         -> No results found (treating as raw score = -10)")
            raw = -10.0
        conf = sigmoid(raw)
        print(f"         -> Raw score: {raw:+.4f}  |  Confidence: {conf*100:.1f}%")
        out_scope_scores.append({"question": q, "raw": raw, "confidence": conf})

    return in_scope_scores, out_scope_scores


# =============================================================================
#  STEP 3: Sweep thresholds and find the best one
# =============================================================================

def sweep_thresholds(in_scope_scores, out_scope_scores):
    thresholds = [round(t, 2) for t in np.arange(0.40, 0.96, 0.01)]

    results = []
    for t in thresholds:
        # In-scope questions SHOULD pass (confidence >= threshold)
        in_correct  = sum(1 for s in in_scope_scores  if s["confidence"] >= t)
        # Out-scope questions SHOULD fail (confidence < threshold)
        out_correct = sum(1 for s in out_scope_scores if s["confidence"] <  t)

        total_correct = in_correct + out_correct
        total         = len(in_scope_scores) + len(out_scope_scores)
        accuracy      = total_correct / total if total > 0 else 0.0

        in_acc  = in_correct  / len(in_scope_scores)  if in_scope_scores  else 0.0
        out_acc = out_correct / len(out_scope_scores) if out_scope_scores else 0.0

        # Balanced score: harmonic mean to penalise if either side is weak
        if in_acc + out_acc > 0:
            balanced = 2 * (in_acc * out_acc) / (in_acc + out_acc)
        else:
            balanced = 0.0

        results.append({
            "threshold": t,
            "accuracy":  accuracy,
            "in_acc":    in_acc,
            "out_acc":   out_acc,
            "balanced":  balanced,
            "in_pass":   in_correct,
            "out_fail":  out_correct,
        })

    return results


# =============================================================================
#  STEP 4: Print the results table and recommend
# =============================================================================

def print_results_table(sweep_results, in_scope_scores, out_scope_scores):
    n_in  = len(in_scope_scores)
    n_out = len(out_scope_scores)

    print("\n\n" + "="*80)
    print("  THRESHOLD SWEEP RESULTS")
    print("="*80)
    print(f"  {'Threshold':>10}  {'In-Scope pass':>13}  {'Out-Scope refused':>17}  {'Balanced':>10}  {'Overall':>9}")
    print(f"  {'----------':>10}  {'-------------':>13}  {'-----------------':>17}  {'----------':>10}  {'---------':>9}")

    best_balanced = max(sweep_results, key=lambda x: x["balanced"])

    for r in sweep_results:
        marker = "  <-- BEST" if r["threshold"] == best_balanced["threshold"] else ""
        print(
            f"  {r['threshold']:>10.2f}"
            f"  {r['in_pass']:>5}/{n_in:<7}"
            f"  {r['out_fail']:>5}/{n_out:<11}"
            f"  {r['balanced']*100:>9.1f}%"
            f"  {r['accuracy']*100:>8.1f}%"
            f"{marker}"
        )

    # Detailed per-question breakdown
    best_t = best_balanced["threshold"]

    print("\n\n" + "="*80)
    print(f"  PER-QUESTION BREAKDOWN  (at recommended threshold = {best_t:.2f})")
    print("="*80)

    print(f"\n  IN-SCOPE questions  (should PASS the threshold)")
    print(f"  " + "-"*70)
    for s in in_scope_scores:
        status = "PASS    " if s["confidence"] >= best_t else "WRONGLY REFUSED"
        print(f"  [{status}]  conf={s['confidence']*100:.1f}%  | {s['question'][:55]}")

    print(f"\n  OUT-OF-SCOPE questions  (should be REFUSED by the threshold)")
    print(f"  " + "-"*70)
    for s in out_scope_scores:
        status = "REFUSED " if s["confidence"] < best_t else "WRONGLY ANSWERED"
        print(f"  [{status}]  conf={s['confidence']*100:.1f}%  | {s['question'][:55]}")

    return best_t


# =============================================================================
#  STEP 5: Save results and optionally update .env
# =============================================================================

def save_calibration_results(sweep_results, in_scope_scores, out_scope_scores, best_threshold):
    os.makedirs("data", exist_ok=True)

    # Convert numpy floats to plain Python floats for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if hasattr(obj, 'item'):  # numpy scalar (float32, int64, etc.)
            return obj.item()
        return obj

    output = {
        "best_threshold":    float(best_threshold),
        "sweep":             to_serializable(sweep_results),
        "in_scope_scores":   to_serializable(in_scope_scores),
        "out_scope_scores":  to_serializable(out_scope_scores),
    }
    path = os.path.join("data", "calibration_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full calibration results saved to: {path}")


def update_env_threshold(best_threshold):
    env_path = ".env"
    if not os.path.exists(env_path):
        print(f"  Warning: .env file not found at {env_path}. Skipping auto-update.")
        return

    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("REFUSAL_THRESHOLD="):
            new_lines.append(f"REFUSAL_THRESHOLD={best_threshold:.2f}\n")
            found = True
        else:
            new_lines.append(line)

    # If key does not exist yet, append it
    if not found:
        new_lines.append(f"\nREFUSAL_THRESHOLD={best_threshold:.2f}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"  Updated REFUSAL_THRESHOLD={best_threshold:.2f} in your .env file")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("RAG Threshold Calibration Tool")
    print("Finds the best REFUSAL_CONFIDENCE_THRESHOLD for your dataset\n")
    print(f"  In-scope questions    : {len(IN_SCOPE_QUESTIONS)}")
    print(f"  Out-of-scope questions: {len(OUT_SCOPE_QUESTIONS)}")

    # Init services
    print("\n  Initializing RAG services...")
    try:
        collection = init_services()
    except Exception as e:
        print(f"  ERROR: Failed to initialize RAG services: {e}")
        print("  Make sure populate_db.py has been run and .env is configured.")
        sys.exit(1)

    bm25, bm25_chunks = get_bm25_index()
    reranker = get_reranker()

    if not reranker.is_active:
        print("  ERROR: No reranker is active. Cannot calibrate without a reranker.")
        print("  Install sentence-transformers: pip install sentence-transformers")
        sys.exit(1)

    print(f"  Reranker provider : {reranker.provider}")
    print(f"  BM25 loaded       : {bm25 is not None}")

    # Collect scores
    in_scope_scores, out_scope_scores = collect_scores(collection, bm25, bm25_chunks, reranker)

    # Sweep thresholds
    print("\n\n  Sweeping thresholds from 0.40 to 0.95...")
    sweep_results = sweep_thresholds(in_scope_scores, out_scope_scores)

    # Print table and get best threshold
    best_threshold = print_results_table(sweep_results, in_scope_scores, out_scope_scores)

    # Summary
    current_threshold = float(os.getenv("REFUSAL_THRESHOLD", "0.70"))
    print("\n\n" + "="*80)
    print("  CALIBRATION SUMMARY")
    print("="*80)
    print(f"  Current threshold (from .env) : {current_threshold:.2f}  ({current_threshold*100:.0f}%)")
    print(f"  Recommended threshold         : {best_threshold:.2f}  ({best_threshold*100:.0f}%)")

    if abs(best_threshold - current_threshold) < 0.01:
        print("  Your current threshold is already optimal!")
    elif best_threshold > current_threshold:
        print(f"  Recommendation: RAISE the threshold (stricter refusals)")
        print(f"  This will reduce wrongly-answered out-of-scope questions.")
    else:
        print(f"  Recommendation: LOWER the threshold (more permissive)")
        print(f"  This will stop refusing valid in-scope questions.")

    # Save results
    save_calibration_results(sweep_results, in_scope_scores, out_scope_scores, best_threshold)

    # Ask to update .env
    print("\n" + "="*80)
    answer = input(f"  Auto-update REFUSAL_THRESHOLD={best_threshold:.2f} in your .env? [y/N]: ").strip().lower()
    if answer == "y":
        update_env_threshold(best_threshold)
    else:
        print(f"\n  To apply manually, add this line to your .env:")
        print(f"  REFUSAL_THRESHOLD={best_threshold:.2f}")

    print("\n  Calibration complete!\n")


if __name__ == "__main__":
    main()
