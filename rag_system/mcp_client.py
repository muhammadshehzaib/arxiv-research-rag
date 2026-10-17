import os
import sys
import json
import asyncio
import time
from typing import List, Dict, Any, Tuple, Optional
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from rag_system.utils import setup_windows_encoding

setup_windows_encoding()

class AcademicMCPClient:
    """
    Client interface to communicate with the Academic Research MCP Server
    via the standard Model Context Protocol (stdio transport).
    Provides fallback to direct tool invocation if subprocess transport is unavailable.
    """
    def __init__(self, server_script: Optional[str] = None):
        self.server_script = server_script

    async def _call_tool_stdio(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script] if self.server_script else ["-m", "rag_system.academic_mcp_server"],
            env=os.environ.copy()
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if result and result.content:
                    text_content = result.content[0].text
                    try:
                        return json.loads(text_content)
                    except json.JSONDecodeError:
                        return {"raw_text": text_content}
                return {"status": "empty_response"}

    def _call_tool_direct(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """In-process fallback in case stdio subprocess communication is interrupted."""
        from rag_system import academic_mcp_server
        if tool_name == "search_arxiv":
            raw = academic_mcp_server.search_arxiv(
                query=arguments.get("query", ""),
                max_results=arguments.get("max_results", 5)
            )
        elif tool_name == "search_semantic_scholar":
            raw = academic_mcp_server.search_semantic_scholar(
                query=arguments.get("query", ""),
                limit=arguments.get("limit", 5)
            )
        elif tool_name == "fetch_paper_by_id":
            raw = academic_mcp_server.fetch_paper_by_id(
                paper_id=arguments.get("paper_id", "")
            )
        elif tool_name == "ingest_live_paper_to_rag":
            raw = academic_mcp_server.ingest_live_paper_to_rag(
                paper_id=arguments.get("paper_id", "")
            )
        else:
            return {"status": "unknown_tool", "tool": tool_name}

        try:
            return json.loads(raw)
        except Exception:
            return {"raw_text": raw}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Synchronous wrapper to execute an MCP tool via stdio with automatic direct fallback."""
        try:
            is_loop_running = False
            try:
                loop = asyncio.get_running_loop()
                is_loop_running = loop.is_running()
            except RuntimeError:
                is_loop_running = False

            if timeout is not None:
                # A bounded web request must never fall through to an unbounded direct call.
                if is_loop_running:
                    raise RuntimeError("Synchronous MCP call cannot run inside an event loop")
                return asyncio.run(asyncio.wait_for(self._call_tool_stdio(tool_name, arguments), timeout=timeout))
            if is_loop_running:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, self._call_tool_stdio(tool_name, arguments)).result(timeout=30)
            else:
                return asyncio.run(self._call_tool_stdio(tool_name, arguments))
        except TimeoutError:
            raise
        except Exception as stdio_err:
            if timeout is not None:
                raise
            print(f"[MCP Client] Notice: Stdio transport ({stdio_err}). Using direct MCP call fallback...")
            return self._call_tool_direct(tool_name, arguments)

    def fetch_live_papers(self, query: str, max_results: int = 4, prefer_source: str = "arxiv", timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Queries the MCP server for live academic research papers.
        Tries primary source first; if empty or rate-limited, falls back to alternative.
        """
        papers = []
        deadline = time.monotonic() + timeout if timeout is not None else None

        def time_left():
            if deadline is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Live paper search exceeded its time limit")
            return remaining

        if prefer_source == "semantic_scholar":
            s2_res = self.call_tool("search_semantic_scholar", {"query": query, "limit": max_results}, timeout=time_left())
            if s2_res.get("status") == "success" and s2_res.get("papers"):
                papers.extend(s2_res["papers"])

        # Fallback or primary arXiv query
        if not papers:
            arxiv_res = self.call_tool("search_arxiv", {"query": query, "max_results": max_results}, timeout=time_left())
            if arxiv_res.get("status") == "success" and arxiv_res.get("papers"):
                papers.extend(arxiv_res["papers"])

        return papers[:max_results]

    def format_live_context(self, papers: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Converts live papers retrieved from MCP into RAG context blocks and source dictionaries.
        """
        context_blocks = []
        sources = []

        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Untitled")
            paper_id = paper.get("paper_id", "Unknown")
            authors_list = paper.get("authors", [])
            authors_str = ", ".join(authors_list) if isinstance(authors_list, list) else str(authors_list)
            published = paper.get("published", "Recent")
            summary = paper.get("summary", "No abstract available.")
            pdf_url = paper.get("pdf_url", "")
            source_label = paper.get("source", "arXiv (Live MCP)")

            block = (
                f"Source [Live {i}]: {title} (ID: {paper_id}, Source: {source_label})\n"
                f"Authors: {authors_str}\n"
                f"Published: {published}\n"
                f"Abstract / Context:\n{summary}\n"
            )
            context_blocks.append(block)

            sources.append({
                "index": i,
                "title": title,
                "authors": authors_str,
                "pdf_url": pdf_url,
                "paper_id": paper_id,
                "relevance_score": None,
                "reranker_provider": None,
                "rrf_score": None,
                "rerank_score": None,
                "text": summary,
                "is_live_mcp": True,
                "source_type": source_label,
                "published": published
            })

        return context_blocks, sources


# Global singleton instance
_mcp_client_instance: Optional[AcademicMCPClient] = None

def get_mcp_client() -> AcademicMCPClient:
    global _mcp_client_instance
    if _mcp_client_instance is None:
        _mcp_client_instance = AcademicMCPClient()
    return _mcp_client_instance

def trigger_background_ingest(paper_id: str):
    """
    Spawns an asynchronous background worker thread to download, chunk,
    and index the live arXiv paper into Chroma DB and SQLite FTS5 index.
    """
    import threading
    client = get_mcp_client()

    def _worker():
        try:
            print(f"\n📥 [AUTO-INGEST WORKER] Downloading and indexing '{paper_id}' into Chroma DB...")
            res = client.call_tool("ingest_live_paper_to_rag", {"paper_id": paper_id})
            if res.get("status") == "success":
                print(f"✨ [AUTO-INGEST WORKER] Paper '{paper_id}' is now permanently indexed in Chroma DB!")
            else:
                print(f"ℹ️ [AUTO-INGEST WORKER] Ingest status for '{paper_id}': {res.get('message', res.get('status'))}")
        except Exception as err:
            print(f"⚠️ [AUTO-INGEST WORKER] Background ingest failed for '{paper_id}': {err}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
