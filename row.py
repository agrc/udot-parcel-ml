#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction
Right of way module containing methods
"""
import logging
import math
import re
from io import BytesIO
from itertools import islice
from os import environ
from pathlib import Path

import cv2
import google.cloud.logging
import google.cloud.storage
import numpy as np
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

if "PY_ENV" in environ and environ["PY_ENV"] == "production":
    client = google.cloud.logging.Client()
    client.setup_logging()

ACCEPTABLE_CHARACTERS = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ:-"
OCR_CONFIG = f"--oem 0 --psm 6 -c tessedit_char_whitelist={ACCEPTABLE_CHARACTERS}"
EXPRESSION = re.compile(r"\s", flags=re.MULTILINE)
URI_TEXTFILE = Path(__file__).parent / "bucket-info" / "udot_bucket_uris.txt"
environ["TESSDATA_PREFIX"] = str(Path(__file__).parent / "training-data")
pytesseract.pytesseract.tesseract_cmd = "tesseract"


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


def convert_pdf_to_jpg_bytes(pdf_as_bytes):
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
    except (TypeError, PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as error:
        logging.error(error)
        messages = error

    count = len(images)

    images = [image.tobytes() for image in images if image is not None]

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))

    #: to calculate circle radius, get input image size
    [height, width, _] = img.shape

    logging.info("dimensions: %d x %d", height, width)

    #: original multiplier of 0.01, bigger seems to work better (0.025)
    multipliers = [
        [0.035, 12],
        [0.015, 12],
        [0.0325, 12],
        [0.0175, 12],
        [0.025, 10],
    ]

    i = 0
    count_down = 5
    circle_count = 0
    detected_circles = None
    inset = 0

    while (circle_count > 100 or circle_count == 0) and count_down > 0:
        i += 1
        logging.info("Run: %8d", i)

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

        logging.info("Circles: %4i", circle_count)
        logging.info("Multiplier: %5.4f", ratio_multiplier)
        logging.info("Fudge: %7i", fudge_value)
        logging.info("Radius: %6d-%d", min_rad, max_rad)
        logging.info("Inset: %6i", inset)

        count_down -= 1

    logging.info("Circles: %4i", circle_count)

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


def get_characters_from_image(image):
    """detect characters in a cv2 image object

    Args:
        image: The cv2 image to detect characters in

    Returns:
        list: A list of detected characters
    """

    #: convert image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    morph = cv2.morphologyEx(grayscale_image, cv2.MORPH_DILATE, structure)

    final_image = cv2.divide(grayscale_image, morph, scale=255)

    #: perform text detection
    result = pytesseract.image_to_string(final_image, config=OCR_CONFIG)

    result = clean_ocr_text(result)

    logging.info('detected characters: "%s"', result)

    return result


def clean_ocr_text(text):
    """clean up OCR text

    Args:
        text (str): The text to clean up

    Returns:
        str: The cleaned up text
    """
    return re.sub(pattern=EXPRESSION, repl="", string=text, count=0)


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


def write_results(frame, obj_name, results):
    """write detected results to a dataframe by concatenating them onto the end of
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


def upload_csv(frame, bucket_name, out_name):
    """upload results dataframe to a GCP bucket as a CSV file

    Args:
        frame (dataframe): dataframe containing the final results
        bucket_name (str): the name of the destination bucket
        out_name (str): the name of the CSV file

    Returns:
        nothing
    """
    logging.info("uploading %s to %s", out_name, bucket_name)
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket_name)
    new_blob = bucket.blob(out_name)
    new_blob.upload_from_string(frame.to_csv(), content_type="text/csv")


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
