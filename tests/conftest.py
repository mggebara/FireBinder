"""Shared test setup: keep every test's output out of the real out/ directory."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_out_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREBINDER_OUT_DIR", str(tmp_path / "out"))
