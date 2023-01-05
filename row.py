#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction
Right of way module containing methods
"""
import logging
import math
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
environ["TESSDATA_PREFIX"] = str(Path(__file__).parent / "training-data")
pytesseract.pytesseract.tesseract_cmd = "tesseract"


def main():
    """doc string"""
    return None


def get_job_files(bucket, page_index, job_size=10, testing=False):
    """gets the blob names from the bucket based on the page index and job size

    bucket: the bucket to get the files from. Use a folder path for testing
    page_index: the index of the page to get the files from
    job_size: the number of files to get for the job
    testing: trick the tool to not use google data and from and to use local file paths"""

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

    pdf_as_bytes: a pdf as bytes

    returns: a list of images and the count of images
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


def circle_detect(item_path, output_path):
    """detect circles in an image and export them to output_path
    item_path: path object of image file
    output_path: path object of output directory for cropped images of detected circles
    returns: nothing; cropped images are exported to output_path
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


def ocr_images(directory):
    """OCRs all image files (.jpg) in directory
    directory: path object of directory in which to OCR all .jpg image files
    returns: dataframe with OCR results
    """
    #: create initial dataframe to work with
    working_df = pd.DataFrame({"Number": [], "Filename": [], "Parcel": []})
    working_df = working_df.astype({"Number": int, "Filename": object, "Parcel": object})

    #: read all files in the directory
    ocr_num = 0
    file_list = list(directory.glob("*.jpg"))
    for ocr_file in file_list:
        ocr_count = ocr_file.stem.rsplit("_", 1)[1]

        logging.debug(f"working on: {ocr_file.stem}")
        ocr_full_path = directory / ocr_file
        #: read in image from bytes
        byte_img = ocr_full_path.read_bytes()
        ocr_img = cv2.imdecode(np.frombuffer(byte_img, dtype=np.uint8), 1)  # 1 means flags=cv2.IMREAD_COLOR

        #: convert image to grayscale
        ocr_gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)

        ocr_text = pytesseract.image_to_string(ocr_gray, config=OCR_CONFIG)
        ocr_text = ocr_text.replace("\n", "").replace("\t", "").replace(" ", "")
        df_new = pd.DataFrame(
            {
                "Number": [int(ocr_count)],
                "Filename": [ocr_file],
                "Parcel": [ocr_text],
            }
        )

        working_df = pd.concat([working_df, df_new], ignore_index=True, sort=False)
        ocr_num += 1

    return working_df


def export_circles_from_image(circles, out_dir, file_path, cv2_image, height, width, inset_distance):
    """export detected circles from an image as jpegs to the out_dir
    circles: list of circles returned from cv2.HoughCircles algorithm
    out_dir: path object of output directory for cropped images of detected circles
    file_path: path object of image file
    cv2_image: cv2 image object (numpy.ndarray)
    height: height of original image
    width: width of original image
    inset_distance: inset distance (pixels) to aid image cropping
    returns: a list of images and the count of images
    """
    #: round the values to the nearest integer
    circles = np.uint16(np.around(circles))

    color = (255, 255, 255)
    thickness = -1

    if out_dir:
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

    for i, data in enumerate(circles[0, :]):  # type: ignore
        #: prepare a black canvas on which to draw circles
        canvas = np.zeros((height, width))
        #: draw a white circle on the canvas where detected:
        center_x = data[0]
        center_y = data[1]

        radius = data[2] - inset_distance  #: inset the radius by number of pixels to remove the circle and noise

        cv2.circle(canvas, (center_x, center_y), radius, color, thickness)

        #: create a copy of the input (3-band image) and mask input to white from the canvas:
        im_copy = cv2_image.copy()
        im_copy[canvas == 0] = (255, 255, 255)

        #: crop image to the roi:
        x = center_x - radius - 20
        y = center_y - radius - 20
        h = 2 * radius + 40
        w = 2 * radius + 40

        cropped_img = im_copy[y : y + h, x : x + w]

        if out_dir:
            original_basename = file_path.stem
            out_file = out_dir / f"{original_basename}_{i}.jpg"
            cv2.imwrite(str(out_file), cropped_img)

        return cropped_img
