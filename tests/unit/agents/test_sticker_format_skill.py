# -*- coding: utf-8 -*-
"""Sanity tests for the ``sticker_format`` skill's CLI.

The CLI is deliberately stdlib + Pillow only (so agents outside
the CoPaw venv can use it).  These tests exercise it as a
subprocess to prove nothing in the skill directory depends on
CoPaw internals.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


SKILL_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "qwenpaw"
    / "agents"
    / "skills"
    / "sticker_format-en"
)
SCRIPT = SKILL_DIR / "scripts" / "prepare_sticker_webp.py"


def _png(path: Path, size=(800, 600), colour=(255, 0, 0, 255)) -> Path:
    Image.new("RGBA", size, colour).save(path)
    return path


@pytest.mark.skipif(
    not SCRIPT.is_file(),
    reason="sticker_format-en skill not installed",
)
def test_cli_default_outputs_sticker_png(tmp_path) -> None:
    """Default CLI run produces PNG (Signal-friendly).  Receivers
    render WebP from non-Signal-Desktop sources as voice messages,
    so PNG is the safer default for stickers."""
    src = _png(tmp_path / "pic.png")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out_path = Path(result.stdout.strip())
    assert out_path.name == "pic.sticker.png"
    assert out_path.is_file()
    with Image.open(out_path) as im:
        assert im.size == (512, 512)
        assert im.format == "PNG"


def test_cli_format_webp_for_whatsapp(tmp_path) -> None:
    """``--format webp`` keeps the WhatsApp-compatible output path."""
    src = _png(tmp_path / "pic.png")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(src), "--format", "webp"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out_path = Path(result.stdout.strip())
    assert out_path.name == "pic.sticker.webp"
    with Image.open(out_path) as im:
        assert im.size == (512, 512)
        assert im.format == "WEBP"


@pytest.mark.skipif(
    not SCRIPT.is_file(),
    reason="sticker_format-en skill not installed",
)
def test_cli_explicit_output_path(tmp_path) -> None:
    src = _png(tmp_path / "pic.png")
    out = tmp_path / "nested" / "custom.webp"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(src),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert result.stdout.strip() == str(out)


@pytest.mark.skipif(
    not SCRIPT.is_file(),
    reason="sticker_format-en skill not installed",
)
def test_cli_missing_input_exits_nonzero(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(tmp_path / "nope.png")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "not found" in result.stderr
