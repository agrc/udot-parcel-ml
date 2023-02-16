#!/usr/bin/env python
# * coding: utf8 *
"""
the file to run to start the project
"""

import logging
from os import environ
from sys import stdout
from time import perf_counter
from types import SimpleNamespace

import row

#: Set up logging
logging.basicConfig(
    stream=stdout,
    format="%(levelname)-7s %(asctime)s %(module)10s:%(lineno)5s %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
)

#: Set up variables
TASK_INDEX = int(environ["CLOUD_RUN_TASK_INDEX"])
TASK_COUNT = int(environ["CLOUD_RUN_TASK_COUNT"])
TOTAL_FILES = int(environ["TOTAL_FILES"])
INDEX = environ["INDEX_FILE_LOCATION"]
BUCKET_NAME = environ["INPUT_BUCKET"]
OUTPUT_BUCKET_NAME = environ["OUTPUT_BUCKET"]
JOB_TYPE = environ["JOB_TYPE"]
JOB_NAME = environ["JOB_NAME"]


def mosaic_all_circles():
    """the main function to execute when cloud run starts the circle detection job"""

    job_start = perf_counter()

    row.mosaic_all_circles(JOB_NAME, BUCKET_NAME, OUTPUT_BUCKET_NAME, INDEX, TASK_INDEX, TASK_COUNT, TOTAL_FILES)

    logging.info(
        "job name: %s task %i: entire job %s",
        JOB_NAME,
        TASK_INDEX,
        row.format_time(perf_counter() - job_start),
    )


def ocr_all_mosaics():
    """the main function to execute when cloud run starts the ocr job"""
    job_start = perf_counter()

    inputs = SimpleNamespace(
        job_name=JOB_NAME,
        input_bucket=BUCKET_NAME,
        output_location=OUTPUT_BUCKET_NAME,
        file_index=INDEX,
        task_index=TASK_INDEX,
        task_count=TASK_COUNT,
        total_size=TOTAL_FILES,
        project_number=int(environ["PROJECT_NUMBER"]),
        processor_id=int(environ["PROCESSOR_ID"]),
    )

    row.ocr_all_mosaics(inputs)

    logging.info(
        "job name: %s task %i: entire job %s",
        JOB_NAME,
        TASK_INDEX,
        row.format_time(perf_counter() - job_start),
    )


if __name__ == "__main__":
    if JOB_TYPE == "mosaic":
        mosaic_all_circles()
    elif JOB_TYPE == "ocr":
        ocr_all_mosaics()
    else:
        logging.error("JOB_TYPE environment variable not set")
