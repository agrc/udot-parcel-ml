#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction
Right of way module containing methods
"""
import logging
import math
import re
from os import environ
from pathlib import Path
from sys import stdout

import cv2
import google.cloud.logging
import google.cloud.storage
import numpy as np
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

logging.basicConfig(
    stream=stdout,
    format="%(levelname)-7s %(asctime)s %(module)10s:%(lineno)5s %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

if "PY_ENV" in environ and environ["PY_ENV"] == "production":
    client = google.cloud.logging.Client()
    client.setup_logging()

ACCEPTABLE_CHARACTERS = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ:-"
OCR_CONFIG = f"--oem 0 --psm 6 -c tessedit_char_whitelist={ACCEPTABLE_CHARACTERS}"
EXPRESSION = re.compile(r"\s", flags=re.MULTILINE)
environ["TESSDATA_PREFIX"] = str(Path(__file__).parent / "training-data")
pytesseract.pytesseract.tesseract_cmd = "tesseract"


def main():
    """doc string"""

    return None


def get_job_files(bucket, page_index, job_size=10, testing=False):
    """gets the blob names from the bucket based on the page index and job size
    Args:
        bucket (str): the bucket to get the files from. Use a folder path for testing
        page_index (number): the index of the page to get the files from
        job_size (number): the number of files to get for the job
        testing (bool): trick the tool to not use google data and from and to use local file paths

    Returns:
        list(str): a list of blob names
    """

    page_index = int(page_index)
    job_size = int(job_size)
    testing = bool(testing)

    if testing is True:
        folder = Path(bucket)

        if not folder.exists():
            raise Exception("folder does not exist")

        return [str(item) for i, item in enumerate(folder.iterdir()) if i < job_size]

    logging.debug("creating storage client")

    storage_client = google.cloud.storage.Client()
    iterator = storage_client.list_blobs(bucket, page_size=job_size, max_results=None, versions=False)

    for page in iterator.pages:
        logging.debug("page %i", iterator.page_number)

        if iterator.page_number < page_index:
            logging.debug("skipping page %i", iterator.page_number)

            continue

        return [blob.name for blob in page]

    return []


def convert_pdf_to_pil(pdf_as_bytes):
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
        logging.debug(error)
        messages = error

    count = len(images)

    return (images, count, messages)


def get_circles(item_path, output_path):
    """detect circles in an image and export them to output_path

    Args:
        item_path (Path): The location of the image to detect circles in
        output_path (Path): The output directory for cropped images of detected circles to be stored

    Returns:
        array: An array of binary images
    """
    if not item_path.name.casefold().endswith(("jpg", "jpeg", "tif", "tiff", "png")):

        return []

    logging.debug("detecting circles in %s", item_path)
    #: read in image from bytes
    byte_img = item_path.read_bytes()
    img = cv2.imdecode(np.frombuffer(byte_img, dtype=np.uint8), 1)  # 1 means flags=cv2.IMREAD_COLOR

    if output_path:
        outfile = str(output_path / "test_1.jpg")  #: checking to make sure the image looks right

        logging.debug(outfile)
        cv2.imwrite(outfile, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))

    #: to calculate circle radius, get input image size
    [height, width, _] = img.shape

    logging.debug("dimensions: %d x %d", height, width)

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
        logging.debug("Run: %8d", i)

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

        logging.debug("Circles: %4i", circle_count)
        logging.debug("Multiplier: %5.4f", ratio_multiplier)
        logging.debug("Fudge: %7i", fudge_value)
        logging.debug("Radius: %6d-%d", min_rad, max_rad)
        logging.debug("Inset: %6i", inset)

        count_down -= 1

    return export_circles_from_image(
        detected_circles,
        output_path,
        item_path,
        img,
        height,
        width,
        inset,
    )


def get_characters(image):
    """detect characters in an image

    Args:
        image (np.array): The image to detect characters in

    Returns:
        list: A list of detected characters
    """
    image_channels = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), 1)  # 1 means flags=cv2.IMREAD_COLOR

    #: convert image to grayscale
    grayscale_image = cv2.cvtColor(image_channels, cv2.COLOR_BGR2GRAY)

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


def export_circles_from_image(circles, out_dir, file_path, cv2_image, height, width, inset_distance):
    """export detected circles from an image as jpegs to the out_dir

        Args:
            circles (array): Circle locations returned from cv2.HoughCircles algorithm
            out_dir (Path): The output directory for cropped images of detected circles
            file_path (Path): The location of the image file
            cv2_image (numpy.ndarray): The image as a numpy array
            height (number): The height of original image
            width (number): The width of original image
            inset_distance (number): The inset distance in pixels to aid image cropping

    Returns:
        list: a list of images and the count of images
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
            original_basename = file_path.stem
            out_file = out_dir / f"{original_basename}_{i}.jpg"
            cv2.imwrite(str(out_file), masked_image)

        masked_images.append(masked_image)

    return masked_images
