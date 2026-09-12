import json
import os

DATA_DIR = os.path.join(".", "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "paper_chunks.json")
PARENTS_PATH = os.path.join(DATA_DIR, "paper_parents.json")

print("📦 Loading existing chunks (this may take a moment)...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    old_chunks = json.load(f)

print(f"   Loaded {len(old_chunks):,} chunks.")

parents = {}
new_children = []
skipped_no_parent = 0

for i, c in enumerate(old_chunks):
    parent_id = c.get("parent_id")
    parent_text = c.get("parent_text")

    if not parent_id or not parent_text:
        # Already migrated? Or malformed? Keep the child as-is.
        skipped_no_parent += 1
        new_children.append(c)
        continue

    # Register parent once
    if parent_id not in parents:
        parents[parent_id] = {
            "parent_id": parent_id,
            "paper_id": c.get("paper_id"),
            "parent_index": c.get("parent_index"),
            "text": parent_text,
            "title": c.get("title"),
            "authors": c.get("authors"),
            "published": c.get("published"),
            "published_int": c.get("published_int"),
            "pdf_url": c.get("pdf_url"),
            "total_pages": c.get("total_pages"),
            "word_count": len(parent_text.split()),
        }

    # Strip parent_text from child
    child = {k: v for k, v in c.items() if k != "parent_text"}
    new_children.append(child)

    if (i + 1) % 50000 == 0:
        print(f"   Processed {i+1:,} / {len(old_chunks):,}...")

print(f"✅ Extracted {len(parents):,} unique parents.")
print(f"✅ Kept {len(new_children):,} children.")
if skipped_no_parent:
    print(f"⚠️ {skipped_no_parent} chunks had no parent_text (already migrated?).")

print("💾 Writing paper_chunks.json (children only)...")
with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
    json.dump(new_children, f, indent=2)

print("💾 Writing paper_parents.json (parents)...")
with open(PARENTS_PATH, "w", encoding="utf-8") as f:
    json.dump(parents, f, indent=2)

print(f"\n📊 Sizes:")
print(f"   Old (backup): {os.path.getsize(CHUNKS_PATH + '.bak') / 1024 / 1024:.1f} MB")
print(f"   New children: {os.path.getsize(CHUNKS_PATH) / 1024 / 1024:.1f} MB")
print(f"   New parents:  {os.path.getsize(PARENTS_PATH) / 1024 / 1024:.1f} MB")
print("\n🎉 Migration complete.")