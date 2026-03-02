import os
import uuid
import json
import asyncio
from pathlib import Path
from agents import Agent, Runner, function_tool
from agents.stream_events import RunItemStreamEvent

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import tool_get_document, tool_get_document_structure, tool_get_page_content

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TOOL USE:
- Call get_document() first to confirm status and page count.
- Call get_document_structure() to find relevant page ranges (use node summaries and start_index/end_index).
- Call get_page_content(pages="5-7") with tight ranges. Never fetch the whole doc.
- For Markdown, pages = line numbers from the structure (the line_num field).
ANSWERING: Answer based only on tool output. Be concise.
"""


class PageIndexClient:
    """
    A client for the PageIndex API.
    Uses an OpenAI Agents SDK agent with 3 tools to answer document questions.
    Flow: Index -> query_agent (tool-use loop) -> Answer
    """
    def __init__(self, api_key: str = None, model: str = "gpt-4o-2024-11-20", workspace: str = None):
        self.api_key = api_key or os.getenv("CHATGPT_API_KEY")
        if self.api_key:
            os.environ["CHATGPT_API_KEY"] = self.api_key
            os.environ["OPENAI_API_KEY"] = self.api_key
        self.model = model
        self.workspace = Path(workspace).expanduser() if workspace else None
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

    def index(self, file_path: str, mode: str = "auto") -> str:
        """Upload and index a document. Returns a document_id."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file_path)[1].lower()

        is_pdf = ext == '.pdf'
        is_md = ext in ['.md', '.markdown']

        if mode == "pdf" or (mode == "auto" and is_pdf):
            print(f"Indexing PDF: {file_path}")
            result = page_index(
                doc=file_path,
                model=self.model,
                if_add_node_summary='yes',
                if_add_node_text='yes',
                if_add_node_id='yes',
                if_add_doc_description='yes'
            )
            self.documents[doc_id] = {
                'id': doc_id,
                'path': file_path,
                'type': 'pdf',
                'structure': result['structure'],
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', '')
            }

        elif mode == "md" or (mode == "auto" and is_md):
            print(f"Indexing Markdown: {file_path}")
            result = asyncio.run(md_to_tree(
                md_path=file_path,
                if_thinning=False,
                if_add_node_summary='yes',
                summary_token_threshold=200,
                model=self.model,
                if_add_doc_description='yes',
                if_add_node_text='yes',
                if_add_node_id='yes'
            ))
            self.documents[doc_id] = {
                'id': doc_id,
                'path': file_path,
                'type': 'md',
                'structure': result['structure'],
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', '')
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        print(f"Indexing complete. Document ID: {doc_id}")
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    def _save_doc(self, doc_id: str):
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.documents[doc_id], f, ensure_ascii=False, indent=2)

    def _load_workspace(self):
        for path in self.workspace.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            doc_id = path.stem
            self.documents[doc_id] = doc
        if self.documents:
            print(f"Loaded {len(self.documents)} document(s) from workspace.")

    # ── Public tool methods (thin wrappers) ───────────────────────────────────

    def get_document(self, doc_id: str) -> str:
        """Return document metadata JSON."""
        return tool_get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """Return document tree structure JSON (without text fields)."""
        return tool_get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content JSON for the given pages string (e.g. '5-7', '3,8', '12')."""
        return tool_get_page_content(self.documents, doc_id, pages)

    # ── Agent core ────────────────────────────────────────────────────────────

    def query_agent(self, doc_id: str, prompt: str, verbose: bool = False) -> str:
        """
        Run the PageIndex agent for a document query.
        The agent automatically calls get_document, get_document_structure,
        and get_page_content tools as needed to answer the question.

        Args:
            verbose: If True, print each tool call and result as they happen.
        """
        client_self = self

        @function_tool
        def get_document() -> str:
            """Get document metadata: status, page count, name, and description."""
            return client_self.get_document(doc_id)

        @function_tool
        def get_document_structure() -> str:
            """Get the document's full tree structure (without text) to find relevant sections."""
            return client_self.get_document_structure(doc_id)

        @function_tool
        def get_page_content(pages: str) -> str:
            """
            Get the text content of specific pages or line numbers.
            Use tight ranges: e.g. '5-7' for pages 5 to 7, '3,8' for pages 3 and 8, '12' for page 12.
            For Markdown documents, use line numbers from the structure's line_num field.
            """
            return client_self.get_page_content(doc_id, pages)

        agent = Agent(
            name="PageIndex",
            instructions=AGENT_SYSTEM_PROMPT,
            tools=[get_document, get_document_structure, get_page_content],
            model=self.model,
        )

        if not verbose:
            result = Runner.run_sync(agent, prompt)
            return result.final_output

        # verbose mode: stream events and print tool calls
        async def _run_verbose():
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

        return asyncio.run(_run_verbose())

    # ── Public query API ──────────────────────────────────────────────────────

    def query(self, doc_id: str, prompt: str) -> str:
        """Ask a question about an indexed document. Returns the agent's answer."""
        return self.query_agent(doc_id, prompt)

    def query_stream(self, doc_id: str, prompt: str):
        """
        Ask a question about an indexed document with streaming output.
        MVP: yields the full answer as a single chunk.
        """
        yield self.query_agent(doc_id, prompt)
