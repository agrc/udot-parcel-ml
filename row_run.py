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
URI_TEXTFILE = Path(__file__).parent / "bucket-info" / "udot_bucket_uris.txt"
UDOT_BUCKET_NAME = "ut-udot-row-county-parcels"
OUTPUT_BUCKET_NAME = "ut-dts-agrc-udot-parcel-recog"
TASK_INDEX = environ["CLOUD_RUN_TASK_INDEX"]
TASK_COUNT = environ["CLOUD_RUN_TASK_COUNT"]

#: Set up main function
def main():
    """the main function to execute when cloud run starts the job"""
    job_start = perf_counter()

    #: Set up results dataframe
    df = pd.DataFrame({"Filename": [], "Parcels": []})
    df = df.astype({"Filename": object, "Parcels": object})

    #: Get files to process for this job
    uri_list = row.get_job_files_from_text_index(UDOT_BUCKET_NAME, TASK_INDEX, TASK_COUNT, testing=False)

    #: Initialize GCP storage client and bucket
    storage_client = google.cloud.storage.Client()
    bucket = bucket = storage_client.bucket(UDOT_BUCKET_NAME)

    #: Iterate over objects to detect circles and perform OCR
    for object_name in files:
        object_start = perf_counter()
        extension = Path(object_name).suffix.casefold()

        if extension == ".pdf":
            conversion_start = perf_counter()
            images, count, messages = row.convert_pdf_to_pil(bucket.blob(object_name).download_as_bytes())
            logging.debug("%s contained %i pages and converted with message %s", object_name, count, messages)
            logging.info(
                "job %i: conversion time taken for object %s: %s",
                TASK_INDEX,
                object_name,
                row.format_time(perf_counter() - conversion_start),
            )

        elif extension in ["jpg", "jpeg", "tif", "tiff", "png"]:
            images = list(bucket.blob(object_name).download_as_bytes())
        else:
            logging.info('Not a valid document or image: "%s"', object_name)

            continue

        #: Process images to get detected circles
        logging.info("detecting circles in %s", object_name)
        all_detected_circles = []
        circle_start = perf_counter()

        for image in images:
            circle_images = row.get_circles_from_image_bytes(image)
            all_detected_circles.extend(circle_images)  #: extend because circle_images will be a list

        logging.info(
            "job %i: circle detection time taken %s: %s",
            TASK_INDEX,
            object_name,
            row.format_time(perf_counter() - circle_start),
        )

        #: Process detected circle images to get detected characters
        logging.info("detecting characters in %s", object_name)
        all_results = []
        character_start = perf_counter()

        for circle in all_detected_circles:
            result_str = row.get_characters_from_image(circle)
            all_results.append(result_str)  #: append because result_str will be a string

        logging.info(
            "job %i: character detection time taken %s: %s",
            TASK_INDEX,
            object_name,
            row.format_time(perf_counter() - character_start),
        )

        logging.info(
            "job %i: total time taken for entire task %s", TASK_INDEX, row.format_time(perf_counter() - object_start)
        )

        result_dataframe = row.write_results(result_dataframe, object_name, all_results)

    #: Upload results to GCP bucket as CSV file
    row.upload_csv(df, OUTPUT_BUCKET_NAME, f"ocr_results_{TASK_INDEX}.csv")

    logging.info("job %i: time taken for entire job %s", TASK_INDEX, row.format_time(perf_counter() - job_start))


if __name__ == "__main__":
    main()
