import os
import sys
import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional
from utils import setup_windows_encoding

setup_windows_encoding()

GRAPH_FILE_PATH = os.path.join("data", "knowledge_graph.json")
METADATA_FILE_PATH = os.path.join("data", "papers_metadata.json")
CHUNKS_FILE_PATH = os.path.join("data", "paper_chunks.json")

class AcademicKnowledgeGraph:
    """
    On-device Academic Knowledge Graph linking Papers, Authors, Categories,
    and Paper-to-Paper Citations with sub-millisecond traversal.
    """
    def __init__(self, graph_file: str = GRAPH_FILE_PATH):
        self.graph_file = graph_file
        self.papers: Dict[str, Dict[str, Any]] = {}
        self.authors: Dict[str, List[str]] = defaultdict(list)
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self.citations_out: Dict[str, List[str]] = defaultdict(list)  # paper -> papers it cites
        self.citations_in: Dict[str, List[str]] = defaultdict(list)   # paper -> papers that cite it
        self.co_authors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.is_loaded = False
        
        if os.path.exists(self.graph_file):
            self.load()

    def build(self, metadata_path: str = METADATA_FILE_PATH, chunks_path: str = CHUNKS_FILE_PATH) -> Dict[str, Any]:
        """
        Builds the entire knowledge graph from metadata and extracted chunk citations.
        """
        print(f"🔄 Building Academic Knowledge Graph from {metadata_path}...")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_papers = json.load(f)

        self.papers = {}
        self.authors = defaultdict(list)
        self.categories = defaultdict(list)
        self.citations_out = defaultdict(list)
        self.citations_in = defaultdict(list)
        self.co_authors = defaultdict(lambda: defaultdict(int))

        # Base ID lookup map: '2506.06962' -> '2506.06962v3'
        base_to_full_id = {}

        for p in raw_papers:
            pid = p["paper_id"]
            base_id = pid.split("v")[0]
            base_to_full_id[base_id] = pid
            
            clean_authors = [a.strip() for a in p.get("authors", []) if a.strip()]
            clean_categories = [c.strip() for c in p.get("categories", []) if c.strip()]
            
            self.papers[pid] = {
                "paper_id": pid,
                "title": p.get("title", ""),
                "published": p.get("published", ""),
                "pdf_url": p.get("pdf_url", ""),
                "authors": clean_authors,
                "categories": clean_categories,
                "summary": p.get("summary", "")[:400] + "..." if len(p.get("summary", "")) > 400 else p.get("summary", "")
            }

            # Map Author -> Papers
            for author in clean_authors:
                self.authors[author].append(pid)

            # Map Co-authorship
            for i, a1 in enumerate(clean_authors):
                for a2 in clean_authors[i+1:]:
                    self.co_authors[a1][a2] += 1
                    self.co_authors[a2][a1] += 1

            # Map Category -> Papers
            for cat in clean_categories:
                self.categories[cat].append(pid)

        # Extract paper-to-paper citations from paper text chunks
        if os.path.exists(chunks_path):
            print(f"📖 Scanning text chunks in {chunks_path} for internal cross-citations...")
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            citation_pairs: Set[tuple] = set()
            arxiv_pattern = re.compile(r'\b(\d{4}\.\d{4,5})\b')

            for chunk in chunks:
                src_full_id = chunk.get("paper_id", "")
                src_base_id = src_full_id.split("v")[0]
                text = chunk.get("text", "")

                for match in arxiv_pattern.findall(text):
                    if match in base_to_full_id:
                        target_full_id = base_to_full_id[match]
                        if target_full_id != src_full_id:
                            citation_pairs.add((src_full_id, target_full_id))

            for src_id, target_id in citation_pairs:
                if target_id not in self.citations_out[src_id]:
                    self.citations_out[src_id].append(target_id)
                if src_id not in self.citations_in[target_id]:
                    self.citations_in[target_id].append(src_id)

        self.is_loaded = True
        self.save()
        return self.stats()

    def save(self):
        """Saves serialized graph representation to disk."""
        os.makedirs(os.path.dirname(self.graph_file), exist_ok=True)
        serializable = {
            "papers": self.papers,
            "authors": dict(self.authors),
            "categories": dict(self.categories),
            "citations_out": dict(self.citations_out),
            "citations_in": dict(self.citations_in),
            "co_authors": {k: dict(v) for k, v in self.co_authors.items()}
        }
        with open(self.graph_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"💾 Knowledge Graph saved to {self.graph_file}")

    def load(self):
        """Loads serialized graph representation into memory."""
        if not os.path.exists(self.graph_file):
            return False
        try:
            with open(self.graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.papers = data.get("papers", {})
            self.authors = defaultdict(list, data.get("authors", {}))
            self.categories = defaultdict(list, data.get("categories", {}))
            self.citations_out = defaultdict(list, data.get("citations_out", {}))
            self.citations_in = defaultdict(list, data.get("citations_in", {}))
            
            raw_co = data.get("co_authors", {})
            self.co_authors = defaultdict(lambda: defaultdict(int))
            for k, v in raw_co.items():
                self.co_authors[k] = defaultdict(int, v)
                
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"⚠️ Failed to load Knowledge Graph from {self.graph_file}: {e}")
            return False

    def get_paper_network(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Traverses all 1-hop and 2-hop graph relationships for a given paper.
        """
        if paper_id not in self.papers:
            # Try finding without version suffix
            base_id = paper_id.split("v")[0]
            matched = [pid for pid in self.papers if pid.startswith(base_id)]
            if matched:
                paper_id = matched[0]
            else:
                return None

        paper = self.papers[paper_id]
        
        # Outgoing citations (papers cited by this paper)
        cites = [
            {"paper_id": cid, "title": self.papers[cid]["title"]}
            for cid in self.citations_out.get(paper_id, [])
            if cid in self.papers
        ]

        # Incoming citations (papers citing this paper)
        cited_by = [
            {"paper_id": cid, "title": self.papers[cid]["title"]}
            for cid in self.citations_in.get(paper_id, [])
            if cid in self.papers
        ]

        # Co-authors of paper's authors
        collaborators = defaultdict(int)
        for a in paper.get("authors", []):
            for co_author, count in self.co_authors.get(a, {}).items():
                if co_author not in paper.get("authors", []):
                    collaborators[co_author] += count

        top_collaborators = sorted(collaborators.items(), key=lambda x: x[1], reverse=True)[:5]

        # Related papers sharing categories & authors
        related = []
        for cat in paper.get("categories", []):
            for related_pid in self.categories.get(cat, [])[:10]:
                if related_pid != paper_id and related_pid in self.papers:
                    related.append({
                        "paper_id": related_pid,
                        "title": self.papers[related_pid]["title"]
                    })
                    if len(related) >= 5:
                        break
            if len(related) >= 5:
                break

        return {
            "paper_id": paper_id,
            "title": paper["title"],
            "published": paper["published"],
            "pdf_url": paper["pdf_url"],
            "authors": paper["authors"],
            "categories": paper["categories"],
            "citations_out_count": len(cites),
            "citations_out": cites,
            "citations_in_count": len(cited_by),
            "cited_by": cited_by,
            "top_collaborators": [{"author": a, "co_authorships": c} for a, c in top_collaborators],
            "related_category_papers": related
        }

    def get_author_network(self, author_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves all publications, collaborators, and fields of expertise for an author.
        """
        target_author = None
        for a in self.authors:
            if a.lower() == author_name.lower() or author_name.lower() in a.lower():
                target_author = a
                break

        if not target_author:
            return None

        paper_ids = self.authors.get(target_author, [])
        author_papers = [
            {"paper_id": pid, "title": self.papers[pid]["title"], "published": self.papers[pid]["published"]}
            for pid in paper_ids
            if pid in self.papers
        ]

        top_collaborators = sorted(
            self.co_authors.get(target_author, {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        categories_count = defaultdict(int)
        for pid in paper_ids:
            if pid in self.papers:
                for cat in self.papers[pid].get("categories", []):
                    categories_count[cat] += 1

        return {
            "author": target_author,
            "total_papers": len(author_papers),
            "papers": author_papers,
            "top_collaborators": [{"author": a, "collaborations": c} for a, c in top_collaborators],
            "primary_categories": sorted(categories_count.items(), key=lambda x: x[1], reverse=True)
        }

    def get_graph_context_for_rag(self, paper_ids: List[str]) -> str:
        """
        Formats relational knowledge graph connections for injection into LLM context.
        """
        if not self.is_loaded:
            return ""

        blocks = []
        for pid in paper_ids:
            network = self.get_paper_network(pid)
            if not network:
                continue

            cites_str = ", ".join([f"'{c['title']}' ({c['paper_id']})" for c in network['citations_out'][:3]])
            cited_str = ", ".join([f"'{c['title']}' ({c['paper_id']})" for c in network['cited_by'][:3]])

            block = (
                f"Knowledge Graph Context for '{network['title']}':\n"
                f"  - Authors: {', '.join(network['authors'])}\n"
                f"  - Categories: {', '.join(network['categories'])}\n"
            )
            if cites_str:
                block += f"  - Directly Cites in Corpus: {cites_str}\n"
            if cited_str:
                block += f"  - Cited By Other Corpus Papers: {cited_str}\n"

            blocks.append(block)

        return "\n".join(blocks)

    def stats(self) -> Dict[str, Any]:
        """Returns statistics of the Knowledge Graph."""
        total_citations = sum(len(v) for v in self.citations_out.values())
        total_coauthorships = sum(len(v) for v in self.co_authors.values()) // 2
        return {
            "total_papers": len(self.papers),
            "total_authors": len(self.authors),
            "total_categories": len(self.categories),
            "total_citations": total_citations,
            "total_coauthor_links": total_coauthorships,
            "graph_file": self.graph_file
        }

# Global singleton
_kg_instance = None
def get_knowledge_graph() -> AcademicKnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = AcademicKnowledgeGraph()
    return _kg_instance

if __name__ == "__main__":
    kg = AcademicKnowledgeGraph()
    if "--build" in sys.argv or not os.path.exists(GRAPH_FILE_PATH):
        stats = kg.build()
        print("\n📊 Knowledge Graph Build Complete:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        stats = kg.stats()
        print("\n📊 Knowledge Graph Loaded:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
