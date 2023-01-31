#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction

Usage:
    row_cli.py storage generate-index (--from=location) [--save-to=location]
    row_cli.py storage pick-range (--from=location --task-index=index --file-count=count --instances=size)
    row_cli.py images process <file_name>
    row_cli.py image convert <file_name> (--save-to=location)
    row_cli.py detect circles <file_name> [--save-to=location]
    row_cli.py detect characters <file_name>
    row_cli.py results write

Options:
    --from=location                 The bucket or directory to operate on
    --task-index=index              The index of the task running
    --instances=size                The number of containers running the job [default: 10]
    --save-to=location    The location to output the stuff
Examples:
    python row_cli.py storage generate-index --from=./test-data --save-to=./data
    python row_cli.py storage pick-range --from=.ephemeral --task-index=0 --instances=10 --file-count=100
    python row_cli.py image convert ./test/data/multiple_page.pdf --save-to=./test
"""

from pathlib import Path

from docopt import docopt

import row


def main():
    """doc string"""
    args = docopt(__doc__, version="1.0")  # type: ignore

    if args["storage"] and args["generate-index"]:
        index = row.generate_index(args["--from"], args["--save-to"])
        print(index)
        print(f"total job size: {len(index)}")

        return

    if args["storage"] and args["pick-range"]:
        jobs = row.get_files_from_index(args["--from"], args["--task-index"], args["--instances"], args["--file-count"])
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

            if args["--save-to"]:
                print(f'saving {count} images to {args["--save-to"]}')

                directory = Path(args["--save-to"])
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
            if args["--save-to"]:
                output_directory = Path(args["--save-to"])

            item_path = Path(args["<file_name>"])

            if not item_path.exists():
                print("file does not exist")

                return

            if not item_path.name.casefold().endswith(("jpg", "jpeg", "tif", "tiff", "png")):
                print("item is incorrect file type")

                return

            return row.get_circles_from_image_bytes(item_path.read_bytes(), output_directory, item_path.name)

        if args["characters"]:
            item_path = Path(args["<file_name>"])

            if not item_path.exists():
                print("file does not exist")

                return

            if not item_path.name.casefold().endswith(("jpg", "jpeg", "tif", "tiff", "png")):
                print("item is incorrect file type")

                return

            print("detecting circles in %s", item_path)

            return row.get_characters(item_path.read_bytes())


if __name__ == "__main__":
    main()
