#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction

Usage:
    row_cli.py storage generate-index (--from=location) [--prefix=prefix --save-to=location]
    row_cli.py storage generate-remaining-index (--full-index=location --processed-index=location) [--save-to=location]
    row_cli.py index filter <file_name>
    row_cli.py storage pick-range (--from=location --task-index=index --file-count=count --instances=size)
    row_cli.py process images --job=name --from=location --save-to=location --index=location --task-index=index --file-count=count --instances=size
    row_cli.py process circles --job=name --from=location --save-to=location --index=location --task-index=index --file-count=count --instances=size --project=number --processor=id
    row_cli.py image convert <file_name> (--save-to=location)
    row_cli.py detect circles <file_name> (--save-to=location) [--mosaic]
    row_cli.py results download <run_name> (--from=location)
    row_cli.py results summarize <run_name> (--from=location)
    row_cli.py ocr-results download <run_name> (--from=location --save-to=location)
    row_cli.py ocr-results clean <file_name> (--save-to=location)
    row_cli.py ocr-results join <file_name> (--save-to=location)
    row_cli.py ocr-results filter <file_name> (--save-to=location)

Options:
    --from=location                 The bucket or directory to operate on
    --task-index=index              The index of the task running
    --instances=size                The number of containers running the job [default: 10]
    --save-to=location              The location to output the stuff
Examples:
    python row_cli.py storage generate-index --from=./test-data --prefix=elephant/mosaics/ --save-to=./data
    python row_cli.py storage generate-remaining-index --full-index=./test-data --processed-index=./test-data --save-to=./data
    python row_cli.py storage pick-range --from=.ephemeral --task-index=0 --instances=10 --file-count=100
    python row_cli.py image convert ./test-data/multiple_page.pdf --save-to=./test
    python row_cli.py detect circles ./test-data/five_circles_with_text.png --save-to=./test --mosaic
    python row_cli.py process images --job=test --from=./test-data --save-to=./.ephemeral --index=./test-data --task-index=0 --file-count=1 --instances=1
    python row_cli.py process circles ---job=test --from=./test-data --save-to=./.ephemeral --index=./test-data --task-index=0 --file-count=1 --instances=1 --project=123456789 --processor=123456789
    python row_cli.py results download bobcat --from=bucket-name
    python row_cli.py ocr-results download alligator --from=bucket-name --save-to=./data
    python row_cli.py ocr-results clean ./data/alligator/combined_ocr_results --save-to=./data
    python row_cli.py ocr-results join ./data/cleaned_ocr_results.csv --save-to=./data
    python row_cli.py ocr-results filter ./data/joined_ocr_results.csv --save-to=./data
"""

import logging
from pathlib import Path
from sys import stdout
from types import SimpleNamespace

from docopt import docopt

import row

logging.basicConfig(
    stream=stdout,
    format="%(levelname)-7s %(asctime)s %(module)10s:%(lineno)5s %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    """doc string"""
    args = docopt(__doc__, version="1.0")  # type: ignore

    if args["storage"] and args["generate-index"]:
        index = row.generate_index(args["--from"], args["--prefix"], args["--save-to"])

        print(index)
        print(f"total job size: {len(index)}")

        return

    if args["storage"] and args["generate-remaining-index"]:
        remaining_index = row.generate_remaining_index(
            args["--full-index"], args["--processed-index"], args["--save-to"]
        )
        print(remaining_index)
        print(f"remaining job size: {len(remaining_index)}")

        return

    if args["storage"] and args["pick-range"]:
        jobs = row.get_files_from_index(args["--from"], args["--task-index"], args["--instances"], args["--file-count"])
        print(jobs)

        return

    if args["image"] and args["convert"]:
        pdf = Path(args["<file_name>"])
        if not pdf.exists():
            print("file does not exist")

            return

        images, count, messages = row.convert_pdf_to_jpg_bytes(pdf.read_bytes(), "cli")
        print(f"{pdf.name} contained {count} pages and converted with message {messages}")

        if args["--save-to"]:
            print(f'saving {count} images to {args["--save-to"]}')

            directory = Path(args["--save-to"])
            if not directory.exists():
                print("directory does not exist")

                return

            for index, image in enumerate(images):
                path = Path(directory / f"{pdf.stem}_{index+1}.jpg")
                path.write_bytes(image)

        return

    if args["detect"] and args["circles"]:
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

        circles = row.get_circles_from_image_bytes(item_path.read_bytes(), output_directory, item_path.name)

        if args["--mosaic"]:
            row.build_mosaic_image(circles, item_path.name, output_directory)

            return

        return circles

    if args["process"] and args["images"]:
        return row.mosaic_all_circles(
            args["--job"],
            args["--from"],
            args["--save-to"],
            args["--index"],
            int(args["--task-index"]),
            int(args["--instances"]),
            int(args["--file-count"]),
        )

    if args["process"] and args["circles"]:
        inputs = SimpleNamespace(
            job_name=args["--job"],
            input_bucket=args["--from"],
            output_location=args["--save-to"],
            file_index=args["--index"],
            task_index=int(args["--task-index"]),
            task_count=int(args["--instances"]),
            total_size=int(args["--file-count"]),
            project_number=int(args["--project"]),
            processor_id=args["--processor"],
        )

        results = row.ocr_all_mosaics(inputs)

        print(f"operation finished with status {results}")

        return

    if args["results"] and args["download"]:
        location = row.download_run(args["--from"], args["<run_name>"])

        print(f"files downloaded to {location}")

    if args["results"] and args["summarize"]:
        row.summarize_run(args["--from"], args["<run_name>"])

    if args["ocr-results"] and args["download"]:
        location = row.download_ocr_results(args["--from"], args["<run_name>"], args["--save-to"])

        print(f"files downloaded to {location}")

    if args["ocr-results"] and args["clean"]:
        location = row.clean_ocr_results(args["<file_name>"], args["--save-to"])

        print(f"cleaned results saved to {location}")

    if args["ocr-results"] and args["join"]:
        location = row.join_spreadsheet_info(args["<file_name>"], args["--save-to"])

        print(f"cleaned and joined results saved to {location}")

    if args["ocr-results"] and args["filter"]:
        location = row.filter_results(args["<file_name>"], args["--save-to"])

        print(f"filtered results saved to {location}")

    if args["index"] and args["filter"]:
        index = Path(args["<file_name>"])
        total_lines = 0
        filtered_lines = 0

        with index.open(mode="r", encoding="utf-8", newline=None) as index_file, index.with_name(
            "filtered_index.txt"
        ).open(mode="w", encoding="utf-8", newline=None) as filtered_index_file:
            for line in index_file:
                total_lines += 1

                item = Path(line.strip())

                if "deed" in line.casefold() or item.suffix.casefold().casefold() not in (
                    ".pdf",
                    ".jpg",
                    ".jpeg",
                    ".tif",
                    ".tiff",
                    ".png",
                ):
                    filtered_lines += 1

                    continue

                filtered_index_file.write(line)

        print(f"total lines: {total_lines} filtered lines: {filtered_lines}")


if __name__ == "__main__":
    main()
