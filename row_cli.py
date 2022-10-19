#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction

Usage:
    row_cli.py storage download (--from-bucket=bucket --task-index=index) [--testing=test]
    row_cli.py images process <file_name>
    row_cli.py image convert <file_name> (--output-directory=directory)
    row_cli.py image rotate <file_name>
    row_cli.py circle prepare <file_name>
    row_cli.py circle detect <prepared_file_name>
    row_cli.py circle crop <file_name>
    row_cli.py ocr prepare <file_name>
    row_cli.py ocr detect <image>
    row_cli.py results write

Options:
    --from-bucket=bucket            The bucket to find the image
    --testing=test                  Trick the tool to not use google data and from and to become
                                        file paths [default: false]
    --task-index=index              The index of the task running
    --output-directory=directory    The location to output the stuff
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
