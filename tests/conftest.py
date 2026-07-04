# Shared test helpers. GPL-3.0-or-later.
from __future__ import annotations

from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def wasp_config_text() -> str:
    return (CONFIGS / "wasp52b.toml").read_text()


def write_toml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.toml"
    p.write_text(text)
    return p
