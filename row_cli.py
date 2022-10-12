#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction
Usage:
    row storage download (--from-bucket=bucket --task-index=index) [--testing=test]
    row images process <file_name>
    row image convert <file_name>
    row image rotate <file_name>
    row circle prepare <file_name>
    row circle detect <prepared_file_name>
    row circle crop <file_name>
    row ocr prepare <file_name>
    row ocr detect <image>
    row results write
Options:
    --from-bucket=bucket    The bucket to find the image
    --testing=test          Trick the tool to not use google data and from and to become file paths [default: false]
    --task-index=index      The index of the task running
Examples:
    row storage download --from-bucket=bucket-name --task-index=0
    row
"""

from docopt import docopt

import row


def main():
    """doc string"""
    args = docopt(__doc__, version="1.0")  # type: ignore
    print(args)


def test():
    """doc string"""
    greeting = row.function()
    print(greeting)

    return greeting
