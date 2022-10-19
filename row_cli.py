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
    row_cli.py storage download --from-bucket=bucket-name --task-index=0
    row_cli.py image convert ./test/data/multiple_page.pdf --output-directory=./test
"""

from pathlib import Path

from docopt import docopt

import row


def main():
    """doc string"""
    args = docopt(__doc__, version="1.0")  # type: ignore

    if args["test"]:
        print("great job dude")
        return

    if args["storage"] and args["download"]:
        return
        # row.download(args["--from-bucket"], args["--task-index"], args["--testing"])

    if args["images"] and args["process"]:
        return

    if args["image"]:
        if args["convert"]:
            pdf = Path(args["<file_name>"])
            if not pdf.exists():
                print("file does not exist")
                return

            images, count, messages = row.convert_pdf_to_pil(pdf.read_bytes())
            print(f"{pdf.name} contained {count} pages and converted with message {messages}")

            if args["--output-directory"]:
                print(f'saving {count} images to {args["--output-directory"]}')

                directory = Path(args["--output-directory"])
                if not directory.exists():
                    print("directory does not exist")
                    return

                for index, image in enumerate(images):
                    image.save(directory / f"{pdf.stem}_{index+1}.jpg")

            return
        if args["rotate"]:
            return


if __name__ == "__main__":
    main()
