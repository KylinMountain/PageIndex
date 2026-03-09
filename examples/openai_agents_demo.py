"""
PageIndex x OpenAI Agents Demo

Demonstrates how to use PageIndexClient with the OpenAI Agents SDK
to build a document QA agent with 3 tools:
  - get_document()
  - get_document_structure()
  - get_page_content()

Requirements:
    pip install openai-agents

Steps:
  1 — Index PDF and inspect tree structure
  2 — Inspect document metadata
  3 — Ask a question (agent auto-calls tools)
  4 — Reload from workspace and verify persistence
"""
import os
import sys
import asyncio
import concurrent.futures
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import Agent, Runner, function_tool
from agents.stream_events import RunItemStreamEvent

from pageindex import PageIndexClient
import pageindex.utils as utils

PDF_URL = "https://arxiv.org/pdf/2501.12948.pdf"
PDF_PATH = "tests/pdfs/deepseek-r1.pdf"
WORKSPACE = "~/.pageindex"

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TOOL USE:
- Call get_document() first to confirm status and page/line count.
- Call get_document_structure() to find relevant page ranges (use node summaries and start_index/end_index).
- Call get_page_content(pages="5-7") with tight ranges. Never fetch the whole doc.
- For Markdown, pages = line numbers from the structure (the line_num field). Use line_count from get_document() as the upper bound.
ANSWERING: Answer based only on tool output. Be concise.
"""


def query_agent(client: PageIndexClient, doc_id: str, prompt: str, verbose: bool = False) -> str:
    """Run a document QA agent using the OpenAI Agents SDK."""

    @function_tool
    def get_document() -> str:
        """Get document metadata: status, page count, name, and description."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure() -> str:
        """Get the document's full tree structure (without text) to find relevant sections."""
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(pages: str) -> str:
        """
        Get the text content of specific pages or line numbers.
        Use tight ranges: e.g. '5-7' for pages 5 to 7, '3,8' for pages 3 and 8, '12' for page 12.
        For Markdown documents, use line numbers from the structure's line_num field.
        """
        return client.get_page_content(doc_id, pages)

    agent = Agent(
        name="PageIndex",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[get_document, get_document_structure, get_page_content],
        model=client.model,
    )

    async def _run():
        if not verbose:
            result = await Runner.run(agent, prompt)
            return result.final_output

        turn = 0
        stream = Runner.run_streamed(agent, prompt)
        async for event in stream.stream_events():
            if not isinstance(event, RunItemStreamEvent):
                continue
            if event.name == "tool_called":
                turn += 1
                raw = event.item.raw_item
                args = getattr(raw, "arguments", "{}")
                print(f"\n[Turn {turn}] → {raw.name}({args})")
            elif event.name == "tool_output":
                output = str(event.item.output)
                preview = output[:200] + "..." if len(output) > 200 else output
                print(f"         ← {preview}")
        return stream.final_output

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        return asyncio.run(_run())


# ── Download PDF if needed ─────────────────────────────────────────────────────
if not os.path.exists(PDF_PATH):
    print(f"Downloading {PDF_URL} ...")
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    with requests.get(PDF_URL, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(PDF_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Download complete.\n")

# ── Setup ──────────────────────────────────────────────────────────────────────
client = PageIndexClient(workspace=WORKSPACE)

# ── Step 1: Index + Tree ───────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Indexing PDF and inspecting tree structure")
print("=" * 60)
doc_id = client.index(PDF_PATH)
print(f"\nDocument ID: {doc_id}")
print("\nTree Structure:")
utils.print_tree(client.documents[doc_id]["structure"])

# ── Step 2: Document Metadata ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Document Metadata (get_document)")
print("=" * 60)
print(client.get_document(doc_id))

# ── Step 3: Agent Query ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Agent Query (auto tool-use)")
print("=" * 60)
question = "What are the main conclusions of this paper?"
print(f"\nQuestion: '{question}'\n")
answer = query_agent(client, doc_id, question)
print("Answer:")
print(answer)

# ── Step 4: Persistence Verification ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Persistence — reload without re-indexing")
print("=" * 60)
client2 = PageIndexClient(workspace=WORKSPACE)
answer2 = query_agent(client2, doc_id, "What are the main conclusions of this paper?", verbose=True)
print("Answer from reloaded client:")
print(answer2)
print("\nPersistence verified. ✓")
