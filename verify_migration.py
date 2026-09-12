import json

with open("data/paper_chunks.json") as f:
    children = json.load(f)

with open("data/paper_parents.json") as f:
    parents = json.load(f)

print("Children:", len(children))
print("Parents:", len(parents))
print("Sample child keys:", list(children[0].keys()))

assert not any("parent_text" in c for c in children), "❌ Still duplicated!"
missing = [c["parent_id"] for c in children[:1000] if c["parent_id"] not in parents]
assert not missing, f"❌ Orphan children found: {missing[:5]}"

print("✅ Migration verified")