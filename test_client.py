"""
PageIndex Local SDK Demo
Mirrors the notebook demo flow:
  Step 1 — Index PDF and inspect tree structure
  Step 2 — Reasoning-based retrieval
  Step 3 — Answer generation (streaming)
"""
from pageindex import PageIndexClient
import pageindex.utils as utils

PDF_URL = "https://arxiv.org/pdf/2501.12948.pdf"
PDF_PATH = "tests/pdfs/deepseek-r1.pdf"
WORKSPACE = "~/.pageindex"

# ── Download PDF if needed ────────────────────────────────────────────────────
import os, requests
if not os.path.exists(PDF_PATH):
    print(f"Downloading {PDF_URL} ...")
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    r = requests.get(PDF_URL)
    with open(PDF_PATH, "wb") as f:
        f.write(r.content)
    print("Download complete.\n")

# ── Setup ────────────────────────────────────────────────────────────────────
client = PageIndexClient(workspace=WORKSPACE)

# ── Step 1: Index ────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Indexing PDF")
print("=" * 60)
doc_id = client.index(PDF_PATH)
print(f"\nDocument ID: {doc_id}")

doc = client.documents[doc_id]
print(f"Document name: {doc.get('doc_name', 'N/A')}")
print(f"Description:   {doc.get('doc_description', 'N/A')}")

print("\nTree Structure:")
utils.print_tree(doc["structure"])

# ── Step 2: Retrieval ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Reasoning-Based Retrieval")
print("=" * 60)

query = "What are the conclusions in this document?"
print(f"\nQuery: '{query}'")

nodes = client.retrieve(doc_id, query)
print(f"\nRetrieved {len(nodes)} node(s):")
for n in nodes:
    page_info = f"  pages {n.get('start_index')}–{n.get('end_index')}" if n.get('start_index') else ""
    print(f"  [{n['node_id']}] {n['title']}{page_info}")

# ── Step 3: Streaming Answer ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Answer Generation (streaming)")
print("=" * 60)

query2 = "What are the conclusions in this document?"
print(f"\nQuery: '{query2}'\n")
for chunk in client.query_stream(doc_id, query2):
    print(chunk, end="", flush=True)
print("\n")

# ── Persistence Demo ──────────────────────────────────────────────────────────
print("=" * 60)
print("Bonus: Persistence — reload without re-indexing")
print("=" * 60)
client2 = PageIndexClient(workspace=WORKSPACE)
nodes2 = client2.retrieve(doc_id, "What are the conclusions in this document?")
print(f"Retrieved {len(nodes2)} node(s) without re-indexing. ✓")
