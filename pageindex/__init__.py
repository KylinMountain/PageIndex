from .page_index import *
from .page_index_md import md_to_tree
from .retrieve import tool_get_document, tool_get_document_structure, tool_get_page_content

try:
    from .client import PageIndexClient
except ImportError as _e:
    _import_error_msg = str(_e)

    class PageIndexClient:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PageIndexClient requires 'openai-agents'. "
                "Install it with: pip install openai-agents\n"
                f"(Original error: {_import_error_msg})"
            )
