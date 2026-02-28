import os
import uuid
import json
import asyncio
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import retrieve as tree_retrieve
from .utils import ChatGPT_API, ChatGPT_API_stream, extract_json, remove_fields, create_node_mapping

class PageIndexClient:
    """
    A client for the PageIndex API, designed to mimic how humans read and understand documents.
    Flow: Index -> Retrieve -> Query
    """
    def __init__(self, api_key: str = None, model: str = "gpt-4o-2024-11-20", workspace: str = None):
        self.api_key = api_key or os.getenv("CHATGPT_API_KEY")
        if self.api_key:
            os.environ["CHATGPT_API_KEY"] = self.api_key
        self.model = model
        self.workspace = Path(workspace).expanduser() if workspace else None
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

    def index(self, file_path: str, mode: str = "auto") -> str:
        """
        Upload and index a document.
        Returns a document_id.
        """
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
                if_add_node_text='yes',  # Keep text for easier retrieval
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
                if_add_node_text='yes',  # Keep text for easier retrieval
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

    def retrieve(self, doc_id: str, prompt: str, top_k: int = 5) -> List[Dict]:
        """
        Find relevant sections within a specific document using tree search reasoning.
        This simulates a human traversing the document's table of contents level-by-level.
        """
        doc_info = self.documents.get(doc_id)
        if not doc_info:
            raise ValueError(f"Document {doc_id} not found. Please index it first.")
        
        pdf_path = doc_info['path'] if doc_info.get('type') == 'pdf' else None

        retrieved_nodes = tree_retrieve(
            query=prompt,
            tree=doc_info['structure'],
            pdf_path=pdf_path,
            model=self.model,
            top_k=top_k
        )
                
        return retrieved_nodes

    def query(self, doc_id: str, prompt: str) -> str:
        """
        Ask a question about an indexed document, fetching context automatically.
        """
        print(f"Retrieving context for query: '{prompt}'...")
        retrieved_context = self.retrieve(doc_id, prompt)
        
        if not retrieved_context:
            return "I couldn't find relevant information in the document to answer your question."
            
        # Synthesize answer
        context_text = "\n\n---\n\n".join(
            [f"Section: {n['title']}\nContent: {n['text']}" for n in retrieved_context]
        )
        
        query_prompt = f"""
You are an expert document analysis assistant.
Answer the question based ONLY on the provided context retrieved from the document.
If the context does not contain the answer, say "I don't have enough information to answer that."

Question: {prompt}

Retrieved Context:
{context_text}

Provide a clear, concise, and accurate answer based on the context above.
"""
        print("Generating answer...")
        answer = ChatGPT_API(self.model, query_prompt)
        return answer

    def query_stream(self, doc_id: str, prompt: str):
        """
        Ask a question about an indexed document with streaming output.
        Returns a generator that yields answer tokens one at a time.
        """
        print(f"Retrieving context for query: '{prompt}'...")
        retrieved_context = self.retrieve(doc_id, prompt)

        if not retrieved_context:
            yield "I couldn't find relevant information in the document to answer your question."
            return

        context_text = "\n\n---\n\n".join(
            [f"Section: {n['title']}\nContent: {n['text']}" for n in retrieved_context]
        )

        query_prompt = f"""
You are an expert document analysis assistant.
Answer the question based ONLY on the provided context retrieved from the document.
If the context does not contain the answer, say "I don't have enough information to answer that."

Question: {prompt}

Retrieved Context:
{context_text}

Provide a clear, concise, and accurate answer based on the context above.
"""
        print("Generating answer...")
        yield from ChatGPT_API_stream(self.model, query_prompt)
