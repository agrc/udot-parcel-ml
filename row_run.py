#!/usr/bin/env python
# * coding: utf8 *
"""
the file to run to start the project
"""

import logging
from os import environ
from pathlib import Path
from sys import stdout
from time import perf_counter

import google.cloud.logging
import google.cloud.storage
import pandas as pd

import row

#: Set up logging
logging.basicConfig(
    stream=stdout,
    format="%(levelname)-7s %(asctime)s %(module)10s:%(lineno)5s %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
)

#: Set up variables
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

    #: Set up results dataframe
    result_dataframe = pd.DataFrame({"Filename": [], "Parcels": []})
    result_dataframe = result_dataframe.astype({"Filename": object, "Parcels": object})

    #: Get files to process for this job
    files = row.get_files_from_index(INDEX, TASK_INDEX, TASK_COUNT, TOTAL_FILES)

    #: Initialize GCP storage client and bucket
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME[5:])

    #: Iterate over objects to detect circles and perform OCR
    for object_name in files:
        object_start = perf_counter()
        object_name = object_name.rstrip()
        extension = Path(object_name).suffix.casefold()

        if extension == ".pdf":
            conversion_start = perf_counter()
            images, count, messages = row.convert_pdf_to_jpg_bytes(
                bucket.blob(object_name).download_as_bytes(), object_name
            )
            logging.info("%s contained %i pages and converted with message %s", object_name, count, messages)
            logging.info(
                "job %i: conversion time taken for object %s: %s",
                TASK_INDEX,
                object_name,
                row.format_time(perf_counter() - conversion_start),
            )

        elif extension in [".jpg", ".jpeg", ".tif", ".tiff", ".png"]:
            images = list([bucket.blob(object_name).download_as_bytes()])
        else:
            logging.info('Not a valid document or image: "%s"', object_name)

            continue

        #: Process images to get detected circles
        logging.info("detecting circles in %s", object_name)
        all_detected_circles = []
        circle_start = perf_counter()

        for image in images:
            circle_images = row.get_circles_from_image_bytes(image, None, object_name)
            all_detected_circles.extend(circle_images)  #: extend because circle_images will be a list

        logging.info(
            "job %i: circle detection time taken %s: %s",
            TASK_INDEX,
            object_name,
            row.format_time(perf_counter() - circle_start),
        )

        #: Process detected circle images into a mosaic
        logging.info("mosaicking images in %s", object_name)
        mosaic_start = perf_counter()

        mosaic = row.build_mosaic_image(all_detected_circles, object_name, None)

        logging.info(
            "job %i: image mosaic time taken %s: %s",
            TASK_INDEX,
            object_name,
            row.format_time(perf_counter() - mosaic_start),
        )

        logging.info(
            "job %i: total time taken for entire task %s", TASK_INDEX, row.format_time(perf_counter() - object_start)
        )

        #: Upload mosaic image to GCP bucket
        if not mosaic:
            logging.info('no mosaic image created or uploaded: "%s"', object_name)

            continue

        row.upload_mosaic(mosaic, OUTPUT_BUCKET_NAME, object_name)

    logging.info("job %i: time taken for entire job %s", TASK_INDEX, row.format_time(perf_counter() - job_start))


if __name__ == "__main__":
    main()
