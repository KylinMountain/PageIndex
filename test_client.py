"""
PageIndex Agent SDK Demo
4-step demo using the 3-tool agent:
  Step 1 — Download, index PDF, and inspect tree structure
  Step 2 — Inspect document metadata via get_document()
  Step 3 — Ask a question (agent auto-calls tools)
  Step 4 — Reload from workspace and verify persistence
"""
import os
import requests
from pageindex import PageIndexClient
import pageindex.utils as utils

PDF_URL = "https://arxiv.org/pdf/2501.12948.pdf"
PDF_PATH = "tests/pdfs/deepseek-r1.pdf"
WORKSPACE = "~/.pageindex"

# ── Download PDF if needed ────────────────────────────────────────────────────
if not os.path.exists(PDF_PATH):
    print(f"Downloading {PDF_URL} ...")
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    r = requests.get(PDF_URL)
    with open(PDF_PATH, "wb") as f:
        f.write(r.content)
    print("Download complete.\n")

# ── Setup ─────────────────────────────────────────────────────────────────────
client = PageIndexClient(workspace=WORKSPACE)

# ── Step 1: Index + Tree ──────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Indexing PDF and inspecting tree structure")
print("=" * 60)
doc_id = client.index(PDF_PATH)
print(f"\nDocument ID: {doc_id}")
print("\nTree Structure:")
utils.print_tree(client.documents[doc_id]["structure"])

# ── Step 2: Document Metadata ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Document Metadata (get_document)")
print("=" * 60)
metadata = client.get_document(doc_id)
print(metadata)

# ── Step 3: Agent Query ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Agent Query (auto tool-use)")
print("=" * 60)
question = "What are the main conclusions of this paper?"
print(f"\nQuestion: '{question}'\n")
answer = client.query_agent(doc_id, question)
print("Answer:")
print(answer)

# ── Step 4: Persistence Verification ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Persistence — reload without re-indexing")
print("=" * 60)
client2 = PageIndexClient(workspace=WORKSPACE)
answer2 = client2.query_agent(doc_id, "What are the main conclusions of this paper?")
print("Answer from reloaded client:")
print(answer2)
print("\nPersistence verified. ✓")
