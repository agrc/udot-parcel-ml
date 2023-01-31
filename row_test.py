#!/usr/bin/env python
# * coding: utf8 *
"""
test_row.py
A module that contains tests for the project module.
"""

from pathlib import Path

import row

root = Path(__file__).parent / "test-data"


def test_convert_pdf_to_pil_single_page_pdf():
    pdf = root / "single_page.PDF"

    images, count, _ = row.convert_pdf_to_jpg_bytes(pdf.read_bytes())

    assert count == 1
    assert images is not None


def test_convert_pdf_to_pil_multi_page_pdf():
    pdf = root / "multiple_page.pdf"

    images, count, _ = row.convert_pdf_to_jpg_bytes(pdf.read_bytes())

    assert count == 5
    assert images is not None


def test_convert_pdf_to_pil_handles_invalid_pdf():
    pdf = root / "invalid.pdf"

    images, count, message = row.convert_pdf_to_jpg_bytes(pdf.read_bytes())

    assert count == 0
    assert images == []
    assert message != ""


def test_convert_pdf_to_pil_handles_empty_bytes():
    images, count, message = row.convert_pdf_to_jpg_bytes(None)

    assert count == 0
    assert images == []
    assert message != ""


def test_get_files_from_index_returns_the_page_size():
    task_index = 0
    task_count = 10
    total_size = 100

    jobs = row.get_files_from_index("test-data", task_index, task_count, total_size)

    assert jobs is not None
    assert len(jobs) == 10
    assert int(jobs[0]) == 1
    assert int(jobs[9]) == 10


def test_get_files_from_index_returns_the_page_size_for_the_second_page():
    task_index = 1
    task_count = 10
    total_size = 100

    jobs = row.get_files_from_index("test-data", task_index, task_count, total_size)

    assert jobs is not None
    assert len(jobs) == 10
    assert int(jobs[0]) == 11
    assert int(jobs[9]) == 20


def test_get_circles_finds_circles():
    image = root / "five_circles_with_text.png"

    circles = row.get_circles_from_image_bytes(image.read_bytes(), None, image.name)

    assert circles is not None

    assert len(circles) == 5


def test_get_circles_saves_images_to_output():
    image = root / "five_circles_with_text.png"
    output = Path(__file__).parent / "data" / "nested"

    row.get_circles_from_image_bytes(image.read_bytes(), output, image.name)

    assert output.exists()

    assert len(list(output.glob("*.jpg"))) == 5


def test_clean_ocr_text_removes_newlines():
    text = """this is a

test"""

    assert row.clean_ocr_text(text) == "thisisatest"

    text = """this is a               test"""
    assert row.clean_ocr_text(text) == "thisisatest"


def test_get_characters_finds_characters():
    for item_path in root.glob("crop_*"):
        file_name = item_path.name

        image_array = row.convert_to_cv2_image(item_path.read_bytes())

        characters = row.get_characters_from_image(image_array)

        expected_characters = file_name.replace("crop_", "").replace("_", ":").replace(".jpg", "")

        assert characters is not None
        assert characters == expected_characters
