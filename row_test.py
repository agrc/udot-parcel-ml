#!/usr/bin/env python
# * coding: utf8 *
"""
test_row.py
A module that contains tests for the project module.
"""

from pathlib import Path

import row

root = Path(__file__).parent


def test_convert_pdf_to_pil_single_page_pdf():
    print(root)
    print(root.parent)

    pdf = root / "test" / "data" / "single_page.pdf"

    images, count, _ = row.convert_pdf_to_pil(pdf.read_bytes())

    assert count == 1
    assert images is not None


def test_convert_pdf_to_pil_multi_page_pdf():
    pdf = root / "test" / "data" / "multiple_page.pdf"

    images, count, _ = row.convert_pdf_to_pil(pdf.read_bytes())

    assert count == 5
    assert images is not None


def test_convert_pdf_to_pil_handles_invalid_pdf():
    pdf = root / "test" / "data" / "invalid.pdf"

    images, count, message = row.convert_pdf_to_pil(pdf.read_bytes())

    assert count == 0
    assert images == []
    assert message != ""


def test_convert_pdf_to_pil_handles_empty_bytes():
    images, count, message = row.convert_pdf_to_pil(None)

    assert count == 0
    assert images == []
    assert message != ""
