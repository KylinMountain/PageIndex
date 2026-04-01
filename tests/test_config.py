# tests/sdk/test_config.py
import pytest
from pageindex.config import ConfigLoader


def test_load_defaults():
    c = ConfigLoader()
    opt = c.load()
    assert opt.model == "gpt-5.4"
    assert opt.retrieve_model == "gpt-5.4"
    assert opt.toc_check_page_num == 20


def test_load_with_overrides():
    c = ConfigLoader()
    opt = c.load({"model": "gpt-5.4", "retrieve_model": "claude-sonnet"})
    assert opt.model == "gpt-5.4"
    assert opt.retrieve_model == "claude-sonnet"


def test_unknown_key_raises():
    c = ConfigLoader()
    with pytest.raises(ValueError, match="Unknown config keys"):
        c.load({"nonexistent_key": "value"})
