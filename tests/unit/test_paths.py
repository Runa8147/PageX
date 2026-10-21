"""Cross-platform path ladder: PAGEX_HOME > source checkout > installed.

These tests pin the rules that make "runs on Windows and Linux" true:

* ``PAGEX_HOME`` turns everything into one portable directory (demo pack),
* a git checkout keeps the old behaviour (``./pagex.toml``, ``./data``,
  ``./models``) so developer workflows and the rest of the suite are stable,
* without either (``pip install pagex``) config/data fall back to the
  platformdirs locations for the current OS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pagex.core import paths
from pagex.core.config import Config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start every test from the neutral (no overrides) state."""
    monkeypatch.delenv("PAGEX_HOME", raising=False)
    monkeypatch.delenv("PAGEX_CONFIG", raising=False)
    paths.refresh()
    yield
    monkeypatch.delenv("PAGEX_HOME", raising=False)
    monkeypatch.delenv("PAGEX_CONFIG", raising=False)
    paths.refresh()


@pytest.fixture()
def portable_home(monkeypatch, tmp_path) -> Path:
    home = tmp_path / "pagex-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PAGEX_HOME", str(home))
    paths.refresh()
    return home


# ---------------------------------------------------------------------------
# mode 1: portable (PAGEX_HOME)
# ---------------------------------------------------------------------------
def test_pagex_home_makes_everything_portable(portable_home):
    assert paths.pagex_home() == portable_home
    assert paths.default_config_path() == portable_home / "pagex.toml"
    assert paths.base_dir() == portable_home
    assert paths.bench_path() == portable_home / "bench.json"
    assert paths.describe()["mode"].startswith("portable")

    cfg = Config.load()
    assert cfg.base == portable_home
    # every relative key from pagex.toml now resolves inside PAGEX_HOME
    assert cfg.db_path == portable_home / "data" / "pagex.db"
    assert cfg.artifacts_dir == portable_home / "data" / "artifacts"
    assert cfg.models_dir == portable_home / "models"


def test_pagex_home_is_expanded_and_absolute(portable_home, monkeypatch):
    monkeypatch.setenv("PAGEX_HOME", str(portable_home))
    paths.refresh()
    assert paths.pagex_home().is_absolute()


def test_portable_samples_win_when_present(portable_home):
    (portable_home / "samples").mkdir()
    assert paths.samples_dir() == portable_home / "samples"


# ---------------------------------------------------------------------------
# mode 2: source checkout (developer / CI)
# ---------------------------------------------------------------------------
def test_checkout_keeps_repo_relative_defaults():
    root = paths.repo_root()
    if root is None:                                # pragma: no cover
        pytest.skip("not running from a source checkout")
    assert (root / "pyproject.toml").exists()
    assert paths.default_config_path() == root / "pagex.toml"
    assert paths.base_dir() == root
    assert paths.describe()["mode"] == "source checkout"

    cfg = Config.load()
    assert cfg.base == root
    assert cfg.db_path == root / "data" / "pagex.db"
    assert str(cfg.db_path).endswith(str(Path("data") / "pagex.db"))


def test_checkout_samples_dir_is_the_repo_corpus():
    if paths.repo_root() is None:                   # pragma: no cover
        pytest.skip("not running from a source checkout")
    assert paths.samples_dir() == paths.repo_root() / "samples"


# ---------------------------------------------------------------------------
# mode 3: installed (no checkout, no PAGEX_HOME) -> platformdirs
# ---------------------------------------------------------------------------
def test_installed_mode_falls_back_to_platformdirs(monkeypatch):
    monkeypatch.setattr(paths, "repo_root", lambda: None)
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    paths.refresh()

    assert paths.describe()["mode"] == "installed"
    assert paths.default_config_path() == paths.user_config_dir() / "pagex.toml"
    assert paths.base_dir() == paths.user_data_dir()
    assert paths.default_config_path().parent == paths.user_config_dir()

    # the OS-specific locations from the plan (Windows / Linux)
    cfg_home = str(paths.user_config_dir()).lower()
    data_home = str(paths.user_data_dir()).lower()
    if os.name == "nt":                             # pragma: no cover
        assert cfg_home.endswith("\\appdata\\roaming\\pagex")
        assert data_home.endswith("\\appdata\\local\\pagex")
    else:                                           # pragma: no cover (Linux CI)
        assert cfg_home.endswith("/pagex") and data_home.endswith("/pagex")


def test_installed_mode_puts_models_under_the_data_dir(monkeypatch):
    monkeypatch.setattr(paths, "repo_root", lambda: None)
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    paths.refresh()
    cfg = Config.load()
    assert cfg.models_dir.parent == paths.user_data_dir()
    assert cfg.base == paths.user_data_dir()


# ---------------------------------------------------------------------------
# mode 4: frozen build
# ---------------------------------------------------------------------------
def test_frozen_mode_never_uses_file_relative_data(monkeypatch):
    monkeypatch.setattr(paths, "repo_root", lambda: None)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "bundle_dir",
                        lambda: Path("/fake/bundle"))   # sys._MEIPASS
    paths.refresh()
    cfg = Config.load()
    assert cfg.base == paths.user_data_dir()        # writable state, not the bundle
    assert not str(cfg.db_path).startswith("/fake/bundle")
    assert paths.describe()["mode"] == "frozen"


# ---------------------------------------------------------------------------
# generic invariants
# ---------------------------------------------------------------------------
def test_explicit_config_resolves_relative_to_its_own_folder(tmp_path):
    conf = tmp_path / "custom" / "pagex.toml"
    conf.parent.mkdir(parents=True)
    conf.write_text('[paths]\ndb = "my.db"\n', encoding="utf-8")
    cfg = Config.load(conf)
    assert cfg.path == conf.resolve()
    assert cfg.base == conf.parent
    assert cfg.db_path == conf.parent / "my.db"


def test_config_save_is_lf_only(tmp_path):
    """Line endings must not depend on the OS (git + TeX artifacts)."""
    conf = tmp_path / "pagex.toml"
    cfg = Config.load(conf)
    cfg.n_threads = 2
    written = cfg.save()
    raw = written.read_bytes()
    assert b"\r" not in raw, "pagex.toml must be written with LF endings"
    assert written.read_text(encoding="utf-8").count("n_threads") == 1
    # and it reads back into the same values
    assert Config.load(written).n_threads == 2


def test_describe_is_json_friendly():
    info = paths.describe()
    assert set(info) >= {"mode", "config", "base_dir", "platform", "python"}
    json.dumps(info)                                 # no Path objects
