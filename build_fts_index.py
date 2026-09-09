import os
import sys
import json
import time
import sqlite3
from utils import setup_windows_encoding, date_to_int

setup_windows_encoding()

FTS_DB_PATH = os.path.join("data", "bm25_fts.db")
CHUNKS_JSON_PATH = os.path.join("data", "paper_chunks.json")

def init_fts_db(db_path=FTS_DB_PATH, rebuild=False):
    """
    Initializes the SQLite FTS5 database schema with WAL mode and synchronization triggers.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if rebuild and os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"🗑️ Removed existing database at {db_path} for rebuild.")
        except Exception as e:
            print(f"⚠️ Could not remove {db_path}: {e}")
        
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # High-performance PRAGMAs for fast read/write and low overhead
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute("PRAGMA cache_size = -64000;") # 64MB cache during build
    
    # Primary relational table for fast metadata lookup
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id TEXT UNIQUE,
        paper_id TEXT,
        title TEXT,
        authors TEXT,
        published TEXT,
        published_int INTEGER,
        total_pages INTEGER,
        chunk_index INTEGER,
        word_count INTEGER,
        parent_id TEXT,
        parent_text TEXT,
        pdf_url TEXT,
        text TEXT
    );
    """)
    
    # Index important metadata columns for instant SQL filtering
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_pub_int ON chunks(published_int);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_pages ON chunks(total_pages);")
    
    # FTS5 Virtual Table for full-text BM25 search
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        title,
        text,
        content='chunks',
        content_rowid='id',
        tokenize='unicode61'
    );
    """)
    
    # Synchronization triggers to ensure the inverted index is always up-to-date
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
    END;
    """)
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, title, text) VALUES('delete', old.id, old.title, old.text);
    END;
    """)
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, title, text) VALUES('delete', old.id, old.title, old.text);
        INSERT INTO chunks_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
    END;
    """)
    
    con.commit()
    return con

def build_index(chunks_path=CHUNKS_JSON_PATH, db_path=FTS_DB_PATH, rebuild=False):
    """
    Reads paper chunks and populates the SQLite FTS5 database in bulk.
    """
    if not os.path.exists(chunks_path):
        print(f"❌ Chunks file not found at {chunks_path}!")
        return False
        
    print(f"🚀 Initializing SQLite FTS5 index build at: {db_path}")
    start_time = time.time()
    
    con = init_fts_db(db_path=db_path, rebuild=rebuild)
    cur = con.cursor()
    
    # Check how many chunks already exist in the database
    cur.execute("SELECT COUNT(*) FROM chunks;")
    existing_count = cur.fetchone()[0]
    print(f"📦 Currently indexed in SQLite: {existing_count} chunks.")
    
    print(f"📖 Loading chunks from {chunks_path}...")
    load_start = time.time()
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks into memory in {time.time() - load_start:.2f}s.")
    
    # Filter for new chunks if not rebuilding
    if not rebuild and existing_count > 0:
        cur.execute("SELECT chunk_id FROM chunks;")
        existing_ids = set(row[0] for row in cur.fetchall())
        chunks_to_insert = [c for c in chunks if c.get("chunk_id") not in existing_ids]
    else:
        chunks_to_insert = chunks
        
    total_to_insert = len(chunks_to_insert)
    print(f"📥 Inserting {total_to_insert} chunks into SQLite FTS5...")
    
    batch_size = 5000
    insert_sql = """
    INSERT OR IGNORE INTO chunks (
        chunk_id, paper_id, title, authors, published, published_int,
        total_pages, chunk_index, word_count, parent_id, parent_text, pdf_url, text
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    insert_start = time.time()
    for i in range(0, total_to_insert, batch_size):
        batch = chunks_to_insert[i : i + batch_size]
        rows = []
        for c in batch:
            authors_str = ", ".join(c["authors"]) if isinstance(c.get("authors"), list) else str(c.get("authors", ""))
            pub_str = c.get("published", "")
            pub_int = c.get("published_int") or date_to_int(pub_str)
            
            rows.append((
                c.get("chunk_id", ""),
                c.get("paper_id", ""),
                c.get("title", ""),
                authors_str,
                pub_str,
                pub_int,
                int(c.get("total_pages", 0)),
                int(c.get("chunk_index", 0)),
                int(c.get("word_count", 0)),
                c.get("parent_id", ""),
                c.get("parent_text", ""),
                c.get("pdf_url", ""),
                c.get("text", "")
            ))
            
        cur.executemany(insert_sql, rows)
        con.commit()
        done = min(i + batch_size, total_to_insert)
        print(f"   Indexed [{done}/{total_to_insert}] chunks ({done / total_to_insert * 100:.1f}%)...")
        
    # Optimize FTS index structure
    print("⚡ Optimizing FTS5 inverted index (merging b-tree segments)...")
    cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize');")
    con.commit()
    
    cur.execute("SELECT COUNT(*) FROM chunks;")
    final_count = cur.fetchone()[0]
    con.close()
    
    total_elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"\n🎉 SQLite FTS5 Indexing Complete!")
    print(f"📊 Total Chunks Indexed: {final_count}")
    print(f"💾 Database File Size:   {file_size_mb:.1f} MB (at {db_path})")
    print(f"⏱️ Total Elapsed Time:   {total_elapsed:.2f}s")
    return True

if __name__ == "__main__":
    rebuild_flag = "--rebuild" in sys.argv
    build_index(rebuild=rebuild_flag)
