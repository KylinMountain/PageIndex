# pageindex/config.py
import yaml
from pathlib import Path
from types import SimpleNamespace


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class ConfigLoader:
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = _DEFAULT_CONFIG_PATH
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> SimpleNamespace:
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, SimpleNamespace):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, SimpleNamespace or None")
        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return SimpleNamespace(**merged)
