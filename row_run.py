#!/usr/bin/env python
# * coding: utf8 *
"""
the file to run to start the project
"""

import logging
from os import environ
from sys import stdout
from time import perf_counter

import row

#: Set up logging
logging.basicConfig(
    stream=stdout,
    format="%(levelname)-7s %(asctime)s %(module)10s:%(lineno)5s %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
)

#: Set up variables
JOB_NAME = environ["JOB_NAME"]
INDEX = environ["INDEX_FILE_LOCATION"]
BUCKET_NAME = environ["INPUT_BUCKET"]
OUTPUT_BUCKET_NAME = environ["OUTPUT_BUCKET"]
TASK_INDEX = int(environ["CLOUD_RUN_TASK_INDEX"])
TASK_COUNT = int(environ["CLOUD_RUN_TASK_COUNT"])
TOTAL_FILES = int(environ["TOTAL_FILES"])

#: Set up main function
def main():
    """the main function to execute when cloud run starts the job"""
    job_start = perf_counter()

    row.process_all(JOB_NAME, BUCKET_NAME, OUTPUT_BUCKET_NAME, INDEX, TASK_INDEX, TASK_COUNT, TOTAL_FILES)

    logging.info("job %i: time taken for entire job %s", TASK_INDEX, row.format_time(perf_counter() - job_start))


if __name__ == "__main__":
    main()
