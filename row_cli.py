#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction

Usage:
    row_cli.py storage get_job_files (--from-bucket=bucket --task-index=index) [--job-size=size --testing=test]
    row_cli.py images process <file_name>
    row_cli.py image convert <file_name> (--output-directory=directory)
    row_cli.py image rotate <file_name>
    row_cli.py circle prepare <file_name>
    row_cli.py circle detect <file_name> (--output-directory=directory)
    row_cli.py circle crop <file_name>
    row_cli.py ocr prepare <file_name>
    row_cli.py ocr detect (--ocr-directory=directory)
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

        if args["circle"] and args["detect"]:
            print("detecting circles ...")

            return

        if args["ocr"] and args["detect"]:
            print("OCRing images ...")

            return


if __name__ == "__main__":
    main()
