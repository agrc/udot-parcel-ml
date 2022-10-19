#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction
Right of way module containing methods
"""
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError


def main():
    """doc string"""
    return None


def function():
    """doc string"""

    return "hi"


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
