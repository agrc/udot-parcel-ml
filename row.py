#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction
Right of way module containing methods
"""
import logging
from os import environ
from pathlib import Path

import google.cloud.logging
import google.cloud.storage
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

if "PY_ENV" in environ and environ["PY_ENV"] == "production":
    client = google.cloud.logging.Client()
    client.setup_logging()


def main():
    """doc string"""
    return None


def function():
    """doc string"""

    return "hi"


def get_job_files(bucket, page_index, job_size=10, testing=False):
    """gets the blob names from the bucket based on the page index and job size

    bucket: the bucket to get the files from. Use a folder path for testing
    page_index: the index of the page to get the files from
    job_size: the number of files to get for the job
    testing: trick the tool to not use google data and from and to use local file paths"""

    page_index = int(page_index)
    job_size = int(job_size)
    testing = bool(testing)

    if testing is True:
        folder = Path(bucket)

        if not folder.exists():
            raise Exception("folder does not exist")

        return [str(item) for i, item in enumerate(folder.iterdir()) if i < job_size]

    logging.debug("creating storage client")

    storage_client = google.cloud.storage.Client()
    iterator = storage_client.list_blobs(bucket, page_size=job_size, max_results=None, versions=False)

    for page in iterator.pages:
        logging.debug("page %i", iterator.page_number)

        if iterator.page_number < page_index:
            logging.debug("skipping page %i", iterator.page_number)

            continue

        return [blob.name for blob in page]

    return []


def convert_pdf_to_pil(pdf_as_bytes):
    """convert pdf to jpg images

    pdf_as_bytes: a pdf as bytes

    returns: a list of images and the count of images
    """
    dpi = 300
    images = []
    messages = ""

    try:
        images = convert_from_bytes(pdf_as_bytes, dpi)
    except (TypeError, PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as error:
        print(error)
        messages = error

    count = len(images)

    return (images, count, messages)
