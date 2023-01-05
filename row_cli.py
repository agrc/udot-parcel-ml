#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction

Usage:
    row_cli.py storage get_job_files (--from-bucket=bucket --task-index=index) [--job-size=size --testing=test]
    row_cli.py images process <file_name>
    row_cli.py image convert <file_name> (--output-directory=directory)
    row_cli.py detect circles <file_name> [--output-directory=directory]
    row_cli.py detect characters <file_name> [--ocr-directory=directory]
    row_cli.py results write

Options:
    --from-bucket=bucket            The bucket to find the image
    --testing=test                  Trick the tool to not use google data and from and to become
                                        file paths [default: False]
    --task-index=index              The index of the task running
    --job-size=size                 The size of the job [default: 10]
    --output-directory=directory    The location to output the stuff
Examples:
    row_cli.py storage download --from-bucket=bucket-name --task-index=0
    row_cli.py image convert ./test/data/multiple_page.pdf --output-directory=./test
"""

from pathlib import Path

import numpy as np
from docopt import docopt

import row


def main():
    """doc string"""
    args = docopt(__doc__, version="1.0")  # type: ignore

    if args["storage"] and args["get_job_files"]:
        jobs = row.get_job_files(args["--from-bucket"], args["--task-index"], args["--job-size"], args["--testing"])
        print(jobs)
        return

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

    if args["detect"]:
        if args["circles"]:
            output_directory = None
            if args["--output-directory"]:
                output_directory = Path(args["--output-directory"])

            return row.get_circles(Path(args["<file_name>"]), output_directory)

        if args["characters"]:
            item_path = Path(args["<file_name>"])

            if not item_path.exists():
                print("file does not exist")

                return

            if not item_path.name.casefold().endswith(("jpg", "jpeg", "tif", "tiff", "png")):
                print("item is incorrect file type")

                return

            image_array = np.frombuffer(item_path.read_bytes(), dtype=np.uint8)
            print("detecting circles in %s", item_path)

            return row.get_characters(image_array)


if __name__ == "__main__":
    main()
