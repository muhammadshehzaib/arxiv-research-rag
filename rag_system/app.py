import os
import json
import asyncio
import time
from functools import partial
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from rag_system.security import guard_api

# Import RAG pipeline from query_rag
from rag_system.query_rag import init_services, query_rag
from rag_system.utils import to_jsonable
from rag_system.query_policy import QUERY_TIMEOUT_SECONDS, quick_reply
app = FastAPI(title="arXiv Research RAG Web UI")

# Same-origin UI needs no CORS. Additional browser origins must be explicit.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("RAG_ALLOWED_ORIGINS", "").split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)
app.middleware("http")(guard_api)

# Global database collection and BM25 search reference
collection = None
bm25 = None
bm25_chunks = None
query_slots = asyncio.Semaphore(4)


@app.get("/api/health")
def get_health():
    """Cheap service check; readiness indicates whether RAG initialized at startup."""
    return {"status": "ok", "rag_ready": collection is not None}

@app.on_event("startup")
def startup_event():
    global collection, bm25, bm25_chunks
    try:
        collection = init_services()
    except Exception as e:
        print(f"❌ Failed to initialize RAG services on startup: {e}")
        
    fts_db_path = os.path.join("data", "bm25_fts.db")
    if os.path.exists(fts_db_path):
        print("⚡ Using disk-backed SQLite FTS5 inverted index for BM25 search (0 MB RAM overhead).")
    else:
        try:
            from rag_system.query_rag import init_bm25
            bm25, bm25_chunks = init_bm25()
            print("✅ BM25 sparse search index initialized on startup.")
        except Exception as e:
            print(f"❌ Failed to initialize BM25 sparse search index: {e}")

class ChatMessage(BaseModel):
    role: Literal["user", "human", "assistant", "ai", "model"]
    content: str = Field(max_length=4000)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    paper_id: Optional[str] = Field(default=None, max_length=80)
    published_after: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    min_pages: Optional[int] = Field(default=None, ge=1, le=10000)
    num_results: int = Field(default=3, ge=1, le=10)
    chat_history: Optional[list[ChatMessage]] = Field(default=None, max_length=12)
    enable_mcp_fallback: Optional[bool] = True

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be blank.")
        return value

    def history_dicts(self):
        return [message.model_dump() for message in self.chat_history] if self.chat_history else None

@app.get("/api/papers")
def get_papers():
    metadata_path = os.path.join("data", "papers_metadata.json")
    if not os.path.exists(metadata_path):
        return []
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail="The request failed. Check server logs.")

@app.get("/api/stats")
def get_stats():
    global collection
    if collection is None:
        try:
            collection = init_services()
        except Exception:
            pass
            
    metadata_path = os.path.join("data", "papers_metadata.json")
    paper_count = 0
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                paper_count = len(json.load(f))
        except Exception:
            pass
            
    chunk_count = 0
    if collection is not None:
        try:
            chunk_count = collection.count()
        except Exception:
            pass
            
    return to_jsonable({"papers": paper_count, "chunks": chunk_count})

@app.post("/api/query")
async def post_query(req: QueryRequest):
    global collection, bm25, bm25_chunks
    # Short conversational requests do not need an initialized database.
    reply = quick_reply(req.query)
    if reply is not None:
        return {"answer": reply, "sources": []}
    request_started = time.monotonic()
    deadline = request_started + QUERY_TIMEOUT_SECONDS
    if collection is None:
        try:
            collection = await asyncio.wait_for(asyncio.to_thread(init_services), timeout=max(0.1, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="RAG initialization took too long. Please try again.")
        except Exception as e:
            raise HTTPException(status_code=500, detail="Research services are unavailable.")

    if query_slots.locked():
        raise HTTPException(status_code=503, detail="Research service is busy. Please try again shortly.")
    await query_slots.acquire()
    try:
        work = partial(query_rag,
            collection,
            req.query,
            num_results=req.num_results,
            paper_id=req.paper_id,
            published_after=req.published_after,
            min_pages=req.min_pages,
            bm25=bm25,
            bm25_chunks=bm25_chunks,
            chat_history=req.history_dicts(),
            enable_mcp_fallback=req.enable_mcp_fallback if req.enable_mcp_fallback is not None else True,
            deadline=deadline,
        )
        future = asyncio.get_running_loop().run_in_executor(None, work)
        future.add_done_callback(lambda completed: query_slots.release())
        answer, sources = await asyncio.wait_for(
            asyncio.shield(future), timeout=max(0.1, deadline - time.monotonic())
        )
        return to_jsonable({"answer": answer, "sources": sources})
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Research query took too long. Please try a more specific question.")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="The request failed. Check server logs.")
    finally:
        print(f"[RAG timing] request_total: {(time.monotonic() - request_started) * 1000:.0f} ms", flush=True)


@app.post("/api/query/stream")
async def stream_query(req: QueryRequest):
    """Send real query stages and the final result as newline-delimited JSON."""
    async def events():
        global collection

        def line(payload):
            return json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n"

        reply = quick_reply(req.query)
        if reply is not None:
            yield line({"type": "result", "answer": reply, "sources": []})
            return

        deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
        yield line({"type": "status", "stage": "local_search"})
        if collection is None:
            try:
                collection = await asyncio.wait_for(
                    asyncio.to_thread(init_services), timeout=max(0.1, deadline - time.monotonic())
                )
            except asyncio.TimeoutError:
                yield line({"type": "error", "message": "The research library took too long to start. Please try again."})
                return
            except Exception:
                yield line({"type": "error", "message": "The research library is unavailable. Please try again shortly."})
                return

        if query_slots.locked():
            yield line({"type": "error", "message": "The research service is busy. Please try again shortly."})
            return

        await query_slots.acquire()
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress(stage):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "status", "stage": stage})

        def finished(future):
            query_slots.release()
            try:
                answer, sources = future.result()
                queue.put_nowait({"type": "result", "answer": answer, "sources": sources})
            except Exception:
                queue.put_nowait({"type": "error", "message": "We couldn't finish that search. Please try again."})

        work = partial(
            query_rag, collection, req.query,
            num_results=req.num_results,
            paper_id=req.paper_id,
            published_after=req.published_after,
            min_pages=req.min_pages,
            bm25=bm25,
            bm25_chunks=bm25_chunks,
            chat_history=req.history_dicts(),
            enable_mcp_fallback=req.enable_mcp_fallback if req.enable_mcp_fallback is not None else True,
            deadline=deadline,
            progress=progress,
        )
        future = loop.run_in_executor(None, work)
        future.add_done_callback(finished)

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=max(0.1, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                yield line({"type": "error", "message": "This search is taking too long. Please try a more specific question."})
                return
            yield line(event)
            if event["type"] in ("result", "error"):
                return

    return StreamingResponse(
        events(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/eval/latest")
def get_eval_latest():
    results_path = os.path.join("data", "eval_results.json")
    if not os.path.exists(results_path):
        return {"status": "no_runs_yet", "message": "No evaluation has been run yet."}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return to_jsonable(verified_evaluation(json.load(f)))
    except Exception as e:
        raise HTTPException(status_code=500, detail="The request failed. Check server logs.")

@app.get("/api/eval/history")
def get_eval_history():
    history_path = os.path.join("data", "eval_history.json")
    if not os.path.exists(history_path):
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return to_jsonable([verified_evaluation(run) for run in json.load(f)])
    except Exception as e:
        raise HTTPException(status_code=500, detail="The request failed. Check server logs.")

def verified_evaluation(run):
    """Old placeholder-based evaluations are not valid baselines."""
    if run.get("schema_version") == 2:
        return run
    result = dict(run, status="legacy_unverified")
    for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "average_score"):
        result[key] = None
    result["details"] = [dict(case, faithfulness=None, answer_relevancy=None,
                             context_precision=None, context_recall=None) for case in run.get("details", [])]
    return result


@app.post("/api/eval/run")
def post_eval_run():
    try:
        from scripts.evaluate_rag import run_evaluation
        results = run_evaluation()
        if results is None:
            raise HTTPException(status_code=500, detail="Evaluation failed to run. Check server logs.")
        return to_jsonable(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail="The request failed. Check server logs.")

@app.get("/api/graph/stats")
def get_graph_stats():
    from rag_system.knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph()
    return kg.stats()

@app.get("/api/graph/paper/{paper_id}")
def get_graph_paper(paper_id: str):
    from rag_system.knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph()
    network = kg.get_paper_network(paper_id)
    if not network:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found in Knowledge Graph")
    return network

@app.get("/api/graph/author/{author_name}")
def get_graph_author(author_name: str):
    from rag_system.knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph()
    network = kg.get_author_network(author_name)
    if not network:
        raise HTTPException(status_code=404, detail=f"Author {author_name} not found in Knowledge Graph")
    return network

# Setup static folders
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

@app.get("/")
def read_root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to arXiv RAG Web UI. static/index.html is missing."}

# Mount static folder for assets (js, css) under /static
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    # Enable environment encoding globally for Windows consoles
    os.environ["PYTHONIOENCODING"] = "utf-8"
    uvicorn.run("rag_system.app:app", host="127.0.0.1", port=8000, reload=True)
