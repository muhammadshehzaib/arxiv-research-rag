import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# Import RAG pipeline from query_rag
from query_rag import init_services, query_rag

app = FastAPI(title="arXiv Research RAG Web UI")

# CORS Setup to allow connections from any client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global database collection reference
collection = None

@app.on_event("startup")
def startup_event():
    global collection
    try:
        collection = init_services()
    except Exception as e:
        print(f"❌ Failed to initialize RAG services on startup: {e}")

class QueryRequest(BaseModel):
    query: str
    paper_id: Optional[str] = None
    published_after: Optional[str] = None
    min_pages: Optional[int] = None
    num_results: Optional[int] = 3

@app.get("/api/papers")
def get_papers():
    metadata_path = os.path.join("data", "papers_metadata.json")
    if not os.path.exists(metadata_path):
        return []
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def post_query(req: QueryRequest):
    global collection
    if collection is None:
        try:
            collection = init_services()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG services not initialized: {e}")
            
    try:
        # Run query through RAG pipeline with optional filters
        answer, sources = query_rag(
            collection,
            req.query,
            num_results=req.num_results,
            paper_id=req.paper_id,
            published_after=req.published_after,
            min_pages=req.min_pages
        )
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Setup static folders
static_dir = os.path.join(os.path.dirname(__file__), "static")
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
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
