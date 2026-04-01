# pageindex/client.py
from __future__ import annotations
import os
from pathlib import Path
from .collection import Collection
from .config import ConfigLoader
from .parser.protocol import DocumentParser


def _normalize_retrieve_model(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """PageIndex client — supports both local and cloud modes.

    Usage:
        # Local mode (auto-detected when no api_key)
        client = PageIndexClient(model="gpt-5.4")

        # Cloud mode (auto-detected when api_key provided)
        client = PageIndexClient(api_key="your-api-key")

        # Or use LocalClient / CloudClient for explicit mode selection
    """

    def __init__(self, api_key: str = None, model: str = None,
                 retrieve_model: str = None, storage_path: str = None,
                 storage=None):
        if api_key:
            self._init_cloud(api_key)
        else:
            self._init_local(model, retrieve_model, storage_path, storage)

    def _init_cloud(self, api_key: str):
        from .backend.cloud import CloudBackend
        self._backend = CloudBackend(api_key=api_key)

    def _init_local(self, model: str = None, retrieve_model: str = None,
                    storage_path: str = None, storage=None):
        if not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")

        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        opt = ConfigLoader().load(overrides or None)

        storage_path = Path(storage_path or "~/.pageindex").expanduser()
        storage_path.mkdir(parents=True, exist_ok=True)

        from .storage.sqlite import SQLiteStorage
        from .backend.local import LocalBackend
        storage_engine = storage or SQLiteStorage(str(storage_path / "pageindex.db"))
        self._backend = LocalBackend(
            storage=storage_engine,
            files_dir=str(storage_path / "files"),
            model=opt.model,
            retrieve_model=_normalize_retrieve_model(opt.retrieve_model or opt.model),
        )

    def collection(self, name: str = "default") -> Collection:
        """Get or create a collection. Defaults to 'default'."""
        self._backend.get_or_create_collection(name)
        return Collection(name=name, backend=self._backend)

    def list_collections(self) -> list[str]:
        return self._backend.list_collections()

    def delete_collection(self, name: str) -> None:
        self._backend.delete_collection(name)

    def register_parser(self, parser: DocumentParser) -> None:
        """Register a custom document parser. Only available in local mode."""
        if not hasattr(self._backend, 'register_parser'):
            from .errors import PageIndexError
            raise PageIndexError("Custom parsers are not supported in cloud mode")
        self._backend.register_parser(parser)


class LocalClient(PageIndexClient):
    """Local mode — indexes and queries documents on your machine."""

    def __init__(self, model: str = None, retrieve_model: str = None,
                 storage_path: str = None, storage=None):
        self._init_local(model, retrieve_model, storage_path, storage)


class CloudClient(PageIndexClient):
    """Cloud mode — fully managed by PageIndex cloud service. No LLM key needed."""

    def __init__(self, api_key: str):
        self._init_cloud(api_key)
