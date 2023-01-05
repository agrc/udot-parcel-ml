#!/usr/bin/env python
# * coding: utf8 *
"""
test_row.py
A module that contains tests for the project module.
"""

from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import row

root = Path(__file__).parent / "test-data"


def test_convert_pdf_to_pil_single_page_pdf():
    pdf = root / "single_page.PDF"

    images, count, _ = row.convert_pdf_to_pil(pdf.read_bytes())

    assert count == 1
    assert images is not None


def test_convert_pdf_to_pil_multi_page_pdf():
    pdf = root / "multiple_page.pdf"

    images, count, _ = row.convert_pdf_to_pil(pdf.read_bytes())

    assert count == 5
    assert images is not None


def test_convert_pdf_to_pil_handles_invalid_pdf():
    pdf = root / "invalid.pdf"

    images, count, message = row.convert_pdf_to_pil(pdf.read_bytes())

    assert count == 0
    assert images == []
    assert message != ""


def test_convert_pdf_to_pil_handles_empty_bytes():
    images, count, message = row.convert_pdf_to_pil(None)

    assert count == 0
    assert images == []
    assert message != ""


@patch("google.cloud.storage.Client")
def test_get_job_files_returns_the_page_size(storage_client):
    job_size = 10
    page_index = 0

    page = iter([Mock() for _ in range(job_size)])

    iterator = Mock()
    iterator.pages = iter([page])
    type(iterator).page_number = PropertyMock(return_value=1)

    client = Mock()
    client.list_blobs = Mock(return_value=iterator)

    storage_client.return_value = client

    jobs = row.get_job_files("bucket", page_index, job_size)

    assert jobs is not None
    assert len(jobs) == 10


@patch("google.cloud.storage.Client")
def test_get_job_files_returns_the_page_size_for_the_second_page(storage_client):
    job_size = 10
    page_index = 2

    blob = Mock()
    type(blob).name = PropertyMock(return_value="page2")

    page_one_blob = Mock()

    page1 = iter([page_one_blob for _ in range(job_size)])
    page2 = iter([blob for _ in range(job_size)])

    iterator = Mock()
    iterator.pages = iter([page1, page2])
    #: logging calls invoke the side effect
    type(iterator).page_number = PropertyMock(side_effect=[1, 1, 1, 2, 2, 2])

    client = Mock()
    client.list_blobs = Mock(return_value=iterator)

    storage_client.return_value = client

    jobs = row.get_job_files("bucket", page_index, job_size)

    assert jobs is not None
    assert len(jobs) == 10
    assert jobs[0] == "page2"
    page_one_blob.assert_not_called()


def test_get_circles_finds_circles():
    image = root / "five_circles_with_text.png"

    circles = row.get_circles(image, None)

    assert circles is not None

    assert len(circles) == 5
