#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction
Right of way module containing methods
"""
import logging
import math
from io import BytesIO
from itertools import islice
from os import environ
from pathlib import Path
from time import perf_counter

import cv2
import google.cloud.logging
import google.cloud.storage
import numpy as np
import pandas as pd
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from PIL import Image
from PIL.Image import DecompressionBombError

if "PY_ENV" in environ and environ["PY_ENV"] == "production":
    client = google.cloud.logging.Client()
    client.setup_logging()


def process_all(job_name, input_bucket, output_location, file_index, task_index, task_count, total_size):
    """the code to run in the cloud run job

    Args:
        job_name (str): the name of the run job. typically named after an animal in alphabetical order
        input_bucket (str): the bucket to get files from using the format `gs://bucket-name`
        output_location (str): the location to save the results to. omit the `gs://` prefix
        file_index (str): the path to the folder containing an `index.txt` file listing all the images in a bucket.
                          `gs://bucket-name`
        task_index (int): the index of the task running
        task_count (int): the number of containers running the job
        total_size (int): the total number of files to process

    Returns:
        None
    """
    #: Get files to process for this job
    files = get_files_from_index(file_index, task_index, task_count, total_size)

    #: Initialize GCP storage client and bucket
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(input_bucket[5:])

    #: Iterate over objects to detect circles and perform OCR
    for object_name in files:
        object_start = perf_counter()
        object_name = object_name.rstrip()
        extension = Path(object_name).suffix.casefold()

        if extension == ".pdf":
            conversion_start = perf_counter()
            images, count, messages = convert_pdf_to_jpg_bytes(
                bucket.blob(object_name).download_as_bytes(), object_name
            )
            logging.info("%s contained %i pages and converted with message %s", object_name, count, messages)
            logging.info(
                "job %i: conversion time taken for object %s: %s",
                task_index,
                object_name,
                format_time(perf_counter() - conversion_start),
            )

        elif extension in [".jpg", ".jpeg", ".tif", ".tiff", ".png"]:
            images = list([bucket.blob(object_name).download_as_bytes()])
        else:
            logging.info('not a valid document or image: "%s"', object_name)

            continue

        #: Process images to get detected circles
        logging.info("detecting circles in %s", object_name)
        all_detected_circles = []
        circle_start = perf_counter()

        for image in images:
            circle_images = get_circles_from_image_bytes(image, None, object_name)
            all_detected_circles.extend(circle_images)  #: extend because circle_images will be a list

        logging.info(
            "job %i: circle detection time taken %s: %s",
            task_index,
            object_name,
            format_time(perf_counter() - circle_start),
        )

        #: Process detected circle images into a mosaic
        logging.info("mosaicking images in %s", object_name)
        mosaic_start = perf_counter()

        mosaic = build_mosaic_image(all_detected_circles, object_name, None)

        logging.info(
            "job %i: image mosaic time taken %s: %s",
            task_index,
            object_name,
            format_time(perf_counter() - mosaic_start),
        )

        logging.info(
            "job %i: total time taken for entire task %s", task_index, format_time(perf_counter() - object_start)
        )

        upload_mosaic(mosaic, output_location, object_name, job_name)


def generate_index(from_location, save_location):
    """reads file names from the `from_location` and optionally saves the list to the `save_location` as an index.txt
    file. Cloud storage buckets must start with `gs://`
    Args:
        from_location (str): the directory to read the files from. Prefix GSC buckets with gs://.
        save_location (str): the directory to save the list of files to. An index.txt file will be created within this
                             directory
    Returns:
        list(str): a list of file names
    """

    files = list([])

    logging.info('reading files from "%s"', from_location)
    if from_location.startswith("gs://"):
        storage_client = google.cloud.storage.Client()
        iterator = storage_client.list_blobs(from_location[5:], max_results=None, versions=False)

        files = [blob.name for blob in iterator]
    else:
        from_location = Path(from_location)

        if not from_location.exists():
            logging.warning("from location %s does not exists", from_location)

            return files

        iterator = from_location.glob("**/*")
        files = [str(item) for item in iterator if item.is_file()]

    if save_location is None:
        return files

    if save_location.startswith("gs://"):
        storage_client = google.cloud.storage.Client()
        bucket = storage_client.bucket(save_location[5:])
        blob = bucket.blob("index.txt")

        with BytesIO() as data:
            for item in files:
                data.write(str(item).encode("utf-8") + b"\n")

            blob.upload_from_string(data.getvalue())
    else:
        save_location = Path(save_location)

        if not save_location.exists():
            logging.warning("save location %s does not exists", save_location)

            return files

        with save_location.joinpath("index.txt").open("w", encoding="utf-8", newline="") as output:
            for item in files:
                output.write(str(item) + "\n")

    return files


def get_files_from_index(from_location, task_index, task_count, total_size):
    """reads the index.txt file from the `from_location`. Based on the task index and total task count a list of files
    is returned. Cloud storage buckets must start with `gs://`
    Args:
        from_location (str): the directory to where the index.txt file resides. Prefix GSC buckets with gs://.
        task_index (number): the index of the current cloud run task
        task_count (number): the total number of cloud run tasks
        total_size (number): the total number of files to process

    Returns:
        list(str): a list of uris from the bucket based on index text file
    """
    task_index = int(task_index)
    task_count = int(task_count)
    total_size = int(total_size)

    index = Path(__file__).parent / ".ephemeral" / "index.txt"

    if from_location.startswith("gs://"):
        storage_client = google.cloud.storage.Client()
        bucket = storage_client.bucket(from_location[5:])
        blob = bucket.blob("index.txt")

        if not index.parent.exists():
            index.parent.mkdir(parents=True)

        try:
            blob.download_to_filename(str(index))
        except Exception as ex:
            logging.error("job %i: error downloading file index %s. %s", task_index, index, ex, exc_info=True)

            raise ex

    else:
        folder = Path(from_location)

        if not folder.exists():
            raise Exception("folder does not exist")

        index = folder.joinpath("index.txt")

        if not index.exists():
            raise Exception("index.txt file does not exist")

    job_size = math.ceil(total_size / task_count)
    first_index = task_index * job_size
    last_index = total_size - 1
    if task_index != (task_count - 1):
        last_index = task_index * job_size + job_size

    file_list = []

    with index.open("r", encoding="utf-8") as data:
        file_list = list(islice(data, first_index, last_index))

    logging.info("calculating files to process for task %i", task_index)

    logging.info("task number %i will work on file indices from %i to %i", task_index, first_index, last_index)

    return file_list


def convert_pdf_to_jpg_bytes(pdf_as_bytes, object_name):
    """convert pdf to jpg images

    Args:
        pdf_as_bytes: a pdf as bytes

    Returns:
        tuple(list, number): A tuple of a list of images and the count of images
    """
    dpi = 300
    images = []
    messages = ""

    try:
        images = convert_from_bytes(pdf_as_bytes, dpi)
    except (TypeError, PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError, DecompressionBombError) as error:
        logging.error("error in %s, %s", object_name, error, exc_info=True)
        messages = error

    count = len(images)

    def convert_to_bytes(image):
        with BytesIO() as byte_array:
            image.save(byte_array, format="JPEG")

            return byte_array.getvalue()

    images = [convert_to_bytes(image) for image in images if image is not None]

    return (images, count, messages)


def get_circles_from_image_bytes(byte_img, output_path, file_name):
    """detect circles in an image (bytes) and export them as a list of cropped images

    Args:
        byte_img (bytes): The image to detect circles in
        output_path (Path): The output directory for cropped images of detected circles to be stored
        file_name (str): The name of the file to be stored
    Returns:
        list: a list of cv2 images
    """

    #: read in image from bytes
    img = cv2.imdecode(np.frombuffer(byte_img, dtype=np.uint8), 1)  # 1 means flags=cv2.IMREAD_COLOR

    if img is None:
        logging.error("unable to read image from bytes: %s", file_name)

        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))

    #: to calculate circle radius, get input image size
    [height, width, _] = img.shape

    #: original multiplier of 0.01, bigger seems to work better (0.025)
    multipliers = [
        [0.010, 12],
        [0.035, 12],
        [0.015, 12],
        [0.0325, 12],
        [0.0175, 12],
        [0.025, 10],
    ]

    i = 0
    count_down = len(multipliers)
    circle_count = 0
    detected_circles = None
    inset = 0

    while (circle_count > 100 or circle_count == 0) and count_down > 0:
        i += 1

        [ratio_multiplier, fudge_value] = multipliers[count_down - 1]

        min_rad = math.ceil(ratio_multiplier * height) - fudge_value
        max_rad = math.ceil(ratio_multiplier * height) + fudge_value

        if min_rad < 15:
            min_rad = 15

        if max_rad < 30:
            max_rad = 30

        #: original inset multiplier of 0.075, bigger seems to work better (0.1)
        inset = int(0.1 * max_rad)

        #: apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(
            image=gray_blur,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_rad,  #: space out circles to prevent multiple detections on the same object
            param1=50,
            param2=50,  #: increased from 30 to 50 to weed out some false circles (seems to work well)
            minRadius=min_rad,
            maxRadius=max_rad,
        )

        if detected_circles is None:
            circle_count = 0
        else:
            circle_count = len(detected_circles[0])

        logging.info(
            "run: %i found %i circles %s",
            i,
            circle_count,
            {
                "multiplier": ratio_multiplier,
                "fudge": fudge_value,
                "diameter": f"{min_rad}-{max_rad}",
                "inset": inset,
                "dimensions": f"{height}x{width}",
            },
        )

        count_down -= 1

    logging.info("final circles count: %i", circle_count)

    return export_circles_from_image(
        detected_circles,
        output_path,
        file_name,
        img,
        height,
        width,
        inset,
    )


def convert_to_cv2_image(image):
    """convert image (bytes) to a cv2 image object

    Args:
        image (bytes): The image (bytes) to convert

    Returns:
        cv2.Image: A cv2 image object
    """
    return cv2.imdecode(np.frombuffer(image, dtype=np.uint8), 1)  # 1 means flags=cv2.IMREAD_COLOR


def export_circles_from_image(circles, out_dir, file_name, cv2_image, height, width, inset_distance):
    """export detected circles from an image as jpegs to the out_dir

        Args:
            circles (array): Circle locations returned from cv2.HoughCircles algorithm
            out_dir (Path): The output directory for cropped images of detected circles
            file_name (str): The name of the image file
            cv2_image (numpy.ndarray): The image as a numpy array
            height (number): The height of original image
            width (number): The width of original image
            inset_distance (number): The inset distance in pixels to aid image cropping

    Returns:
        list: a list of cv2 images
    """
    if circles is None:
        logging.warning("no circles detected for %s", file_name)

        return []

    #: round the values to the nearest integer
    circles = np.uint16(np.around(circles))

    color = (255, 255, 255)
    thickness = -1

    if out_dir:
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

    masked_images = []

    for i, data in enumerate(circles[0, :]):  # type: ignore
        # #: prepare a black canvas on which to draw circles
        canvas = np.zeros((height, width))
        #: draw a white circle on the canvas where detected:
        center_x = data[0]
        center_y = data[1]

        radius = data[2] - inset_distance  #: inset the radius by number of pixels to remove the circle and noise

        cv2.circle(canvas, (center_x, center_y), radius, color, thickness)

        #: create a copy of the input (3-band image) and mask input to white from the canvas:
        image_copy = cv2_image.copy()
        image_copy[canvas == 0] = (255, 255, 255)

        #: crop image to the roi:
        crop_x = center_x - radius - 20
        crop_y = center_y - radius - 20
        crop_height = 2 * radius + 40
        crop_width = 2 * radius + 40

        #: outside of original left edge of image
        if crop_x < 0:
            crop_x = 0
        #: outside of original right edge of image
        if crop_x > width:
            crop_x = width

        #: outside of original top edge of image
        if crop_y < 0:
            crop_y = 0
        #: outside of original bottom edge of image
        if crop_y > height:
            crop_y = height

        masked_image = image_copy[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]

        if out_dir:
            original_basename = Path(file_name).stem
            out_file = out_dir / f"{original_basename}_{i}.jpg"
            cv2.imwrite(str(out_file), masked_image)

        masked_images.append(masked_image)

    return masked_images


def append_results(frame, obj_name, results):
    """append detected results to a dataframe by concatenating them onto the end of
        the existing dataframe

    Args:
        frame (dataframe): dataframe containing the existing results
        obj_name (str): the name of the object (example.pdf, example.jpg, etc.) being processes
        results (list): list of strings containing OCR results from the current file

    Returns:
        dataframe: an updated version of the input dataframe
    """
    df_new = pd.DataFrame({"Filename": [obj_name], "Parcels": [results]})
    frame = pd.concat([frame, df_new], ignore_index=True, sort=False)

    return frame


def upload_results(frame, bucket_name, out_name):
    """upload results dataframe to a GCP bucket as a gzip file

    Args:
        frame (dataframe): dataframe containing the final results
        bucket_name (str): the name of the destination bucket
        out_name (str): the name of the gzip file

    Returns:
        nothing
    """
    file_name = f"{environ['JOB_NAME']}/{out_name}"
    logging.info("uploading %s to %s/%s", out_name, bucket_name, file_name)

    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket_name)
    new_blob = bucket.blob(file_name)

    with BytesIO() as parquet:
        frame.to_parquet(parquet, compression="gzip")

        new_blob.upload_from_file(parquet, content_type="application/gzip")


def format_time(seconds):
    """seconds: number
    returns a human-friendly string describing the amount of time
    """
    minute = 60.00
    hour = 60.00 * minute

    if seconds < 30:
        return "{} ms".format(int(seconds * 1000))

    if seconds < 90:
        return "{} seconds".format(round(seconds, 2))

    if seconds < 90 * minute:
        return "{} minutes".format(round(seconds / minute, 2))

    return "{} hours".format(round(seconds / hour, 2))


def download_run(bucket, run_name):
    """download a runs worth of results from a GCP bucket

    Args:
        bucket (str): the name of the bucket
        run_name (str): the name of the run

    Returns:
        str: the location of the files
    """
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket)
    blobs = bucket.list_blobs(prefix=run_name)
    location = Path(__file__).parent / "data"

    if not location.joinpath(run_name).exists():
        location.joinpath(run_name).mkdir(parents=True)

    for blob in blobs:
        if blob.name.endswith(".gzip"):
            blob.download_to_filename(location / blob.name)

    return location.joinpath(run_name)


def summarize_run(folder, run_name):
    """summarize the results of a run

    Args:
        folder (str): the name of the folder containing the merged results
        run_name (str): the name of the output file

    Returns:
        nothing
    """
    logging.info("summarizing %s", run_name)

    folder = Path(folder) / run_name


def build_mosaic_image(images, object_name, out_dir):
    """build a mosaic image from a list of cv2 images

    Args:
        images (list): list of cv2 images to mosaic together
        object_name (str): the name of the image object (original filename)
        out_dir (Path): location to save the result

    Returns:
        mosaic_image (np.ndarray): composite mosaic of smaller images
    """
    if images is None or len(images) == 0:
        logging.warning("no images to mosaic")

        return np.array(None)

    object_path = Path(object_name)
    max_width = 0
    buffer = 5

    #: Loop through all images to get sizes, save largest
    for img in images:
        if img.shape[1] > max_width:
            max_width = img.shape[1]

    #: Set up parameters for mosaic, calculate number of cols/rows
    number_images = len(images)
    number_columns = math.floor(math.sqrt(number_images))
    number_rows = math.ceil(number_images / number_columns)

    logging.info("mosaicking %i images into %i column by %i row grid", number_images, number_columns, number_rows)

    #: Build mosaic image with white background
    tile_width = max_width + 2 * buffer
    total_height = tile_width * number_rows
    total_width = tile_width * number_columns
    mosaic_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    mosaic_image[:, :] = (255, 255, 255)

    if total_height * total_width > 40_000_000:
        logging.error('mosaic image size is too large: "%s"', object_name)

        return np.array(None)

    i = 0
    for img in images:
        #: Resize all images by inserting them into the same template tile size
        [img_height, img_width, _] = img.shape
        buffered_image = np.zeros((tile_width, tile_width, 3), np.uint8)
        buffered_image[:, :] = (255, 255, 255)
        buffered_image[buffer : buffer + img_height, buffer : buffer + img_width] = img.copy()

        #: Add buffered image into the mosaic
        row_start = (math.floor(i / number_columns)) * tile_width
        col_start = (i % number_columns) * tile_width
        mosaic_image[row_start : row_start + tile_width, col_start : col_start + tile_width] = buffered_image

        i += 1

    if out_dir:
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        mosaic_outfile = out_dir / f"{object_path.stem}.jpg"
        logging.info("saving to %s", mosaic_outfile)
        cv2.imwrite(str(mosaic_outfile), mosaic_image)

    else:
        return mosaic_image


def upload_mosaic(image, bucket_name, object_name):
    """upload mosaic image to a GCP bucket as a jpeg mime type

    Args:
        image (byte-encoded image): the mosaic image to as a numpy array
        bucket_name (str): the name of the destination bucket
        object_name (str): the name of the image object (original filename)

    Returns:
        bool: True if successful, False otherwise
    """
    #: Upload mosaic image to GCP bucket
    if image is None or not image.any():
        logging.info('no mosaic image created or uploaded: "%s"', object_name)

        return False

    object_name = Path(object_name)
    file_name = "mosaics" / object_name
    logging.info("uploading %s to %s/%s", object_name, bucket_name, file_name)

    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket_name)
    new_blob = bucket.blob(file_name)

    #: Encode image
    is_success, buffer = cv2.imencode(".jpg", image)

    if not is_success:
        logging.error("unable to encode image: %s", object_name)

        return False

    with BytesIO(buffer) as data:
        new_blob.upload_from_file(data.getvalue(), content_type="image/jpeg")

    return True
