import os
import json
import asyncio
import logging

try:
    from .utils import (
        ChatGPT_API_async, extract_json,
        get_page_tokens, get_text_of_pdf_pages,
    )
except ImportError:
    from utils import (
        ChatGPT_API_async, extract_json,
        get_page_tokens, get_text_of_pdf_pages,
    )


def retrieve(query, tree, pdf_path=None, model=None, top_k=5):
    """Retrieve relevant document sections for a query using LLM-based tree search.

    Navigates the PageIndex tree level by level: at each level an LLM decides
    which nodes directly answer the query and which need deeper exploration.
    Sibling branches are explored in parallel.

    Args:
        query (str): The question or search query.
        tree (list): PageIndex tree structure (output of page_index or md_to_tree).
        pdf_path (str, optional): Path to the source PDF. If provided, extracts text.
        model (str, optional): LLM model name. Defaults to "gpt-4o-2024-11-20".
        top_k (int, optional): Maximum number of nodes to return.

    Returns:
        list[dict]: Matched nodes, each with keys:
            - node_id (str)
            - title (str)
            - start_index (int)
            - end_index (int)
            - text (str)
    """
    if model is None:
        model = "gpt-4o-2024-11-20"

    pdf_pages = None
    if pdf_path and os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
        pdf_pages = get_page_tokens(pdf_path)

    seen_ids = set()
    results = []

    async def search_level(nodes):
        level_view = [
            {
                'node_id': n['node_id'],
                'title': n.get('title', ''),
                'summary': n.get('summary') or n.get('prefix_summary', ''),
            }
            for n in nodes if n.get('node_id')
        ]
        if not level_view:
            return

        prompt = f"""You are given a question and a list of document sections at the same level of a table of contents.
Each section has a node_id, title, and summary.
Classify each section as:
- "relevant": this section directly contains the answer to the question
- "explore": the answer is likely in one of this section's sub-sections
Omit sections that are clearly unrelated to the question.

Question: {query}

Sections:
{json.dumps(level_view, indent=2)}

Reply in the following JSON format:
{{
    "thinking": "<your reasoning>",
    "relevant": ["node_id_1", ...],
    "explore": ["node_id_2", ...]
}}
Directly return the final JSON structure. Do not output anything else."""

        response = await ChatGPT_API_async(model=model, prompt=prompt)
        result_json = extract_json(response)
        if not result_json:
            logging.warning('retrieve: failed to parse LLM response at this level, skipping')
            return

        node_map = {n['node_id']: n for n in nodes if n.get('node_id')}

        def _get_text(node):
            text = node.get('text', '')
            if not text and pdf_pages and node.get('start_index') is not None and node.get('end_index') is not None:
                text = get_text_of_pdf_pages(pdf_pages, node['start_index'], node['end_index'])
            return text

        def _make_result(node):
            return {
                'node_id': node['node_id'],
                'title': node.get('title', ''),
                'start_index': node.get('start_index'),
                'end_index': node.get('end_index'),
                'line_num': node.get('line_num'),
                'text': _get_text(node),
            }

        for node_id in result_json.get('relevant', []):
            if node_id in seen_ids or node_id not in node_map:
                continue
            seen_ids.add(node_id)
            results.append(_make_result(node_map[node_id]))

        deeper = []
        for node_id in result_json.get('explore', []):
            if node_id in seen_ids or node_id not in node_map:
                continue
            seen_ids.add(node_id)
            node = node_map[node_id]
            children = node.get('nodes', [])
            if children:
                deeper.append(search_level(children))
            else:
                results.append(_make_result(node))

        if deeper:
            await asyncio.gather(*deeper)

    # Handle running inside an existing event loop (e.g. Jupyter)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(search_level(tree))
            new_loop.close()
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
    else:
        asyncio.run(search_level(tree))

    return results[:top_k]
