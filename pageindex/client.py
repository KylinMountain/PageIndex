# pageindex/client.py
from __future__ import annotations
from pathlib import Path
from .collection import Collection
from .config import IndexConfig
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

    @staticmethod
    def _check_llm_api_key(model: str) -> None:
        """Verify that the LLM provider's API key is configured."""
        import os
        try:
            import litellm
            _, provider, _, _ = litellm.get_llm_provider(model=model)
        except Exception:
            return  # Can't resolve provider — let litellm fail later with details

        provider_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "replicate": "REPLICATE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }
        env_var = provider_env.get(provider)
        if env_var and not os.getenv(env_var):
            from .errors import PageIndexError
            raise PageIndexError(
                f"API key not found. Set the {env_var} environment variable "
                f"for provider '{provider}' (model: {model})."
            )

    def _init_local(self, model: str = None, retrieve_model: str = None,
                    storage_path: str = None, storage=None,
                    index_config: IndexConfig | dict = None):
        # Build IndexConfig: merge model/retrieve_model with index_config
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        if isinstance(index_config, IndexConfig):
            opt = index_config.model_copy(update=overrides)
        elif isinstance(index_config, dict):
            overrides.update(index_config)
            opt = IndexConfig(**overrides)
        else:
            opt = IndexConfig(**overrides) if overrides else IndexConfig()

        # Early validation: check API key before any expensive operations
        self._check_llm_api_key(opt.model)

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
            index_config=opt,
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
    """Local mode — indexes and queries documents on your machine.

    Args:
        model: LLM model for indexing (default: gpt-4o-2024-11-20)
        retrieve_model: LLM model for agent QA (default: same as model)
        storage_path: Directory for SQLite DB and files (default: ~/.pageindex)
        storage: Custom StorageEngine instance (default: SQLiteStorage)
        index_config: Advanced indexing parameters. Pass an IndexConfig instance
            or a dict. All fields have sensible defaults — most users don't need this.

    Example::

        # Simple — defaults are fine
        client = LocalClient(model="gpt-5.4")

        # Advanced — tune indexing parameters
        from pageindex.config import IndexConfig
        client = LocalClient(
            model="gpt-5.4",
            index_config=IndexConfig(toc_check_page_num=30),
        )
    """

    def __init__(self, model: str = None, retrieve_model: str = None,
                 storage_path: str = None, storage=None,
                 index_config: IndexConfig | dict = None):
        self._init_local(model, retrieve_model, storage_path, storage, index_config)


class CloudClient(PageIndexClient):
    """Cloud mode — fully managed by PageIndex cloud service. No LLM key needed."""

    def __init__(self, api_key: str):
        self._init_cloud(api_key)
