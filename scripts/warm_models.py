"""Download local query models before the Docker web server accepts requests."""

import os


def main():
    if os.getenv("EMBEDDING_PROVIDER", "gemini") == "local":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        print("Preparing the local embedding model...", flush=True)
        DefaultEmbeddingFunction()(["startup check"])

    from flashrank import Ranker

    print("Preparing the reranking model...", flush=True)
    options = {"model_name": os.getenv("FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2")}
    if cache_dir := os.getenv("FLASHRANK_CACHE_DIR"):
        options["cache_dir"] = cache_dir
    Ranker(**options)
    print("Query models are ready.", flush=True)


if __name__ == "__main__":
    main()
