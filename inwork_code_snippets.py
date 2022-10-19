# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:08:12 2022

@author: eneemann
"""

import math
import os
import time
from pathlib import Path

import cv2
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import Output

os.environ[
    "TESSDATA_PREFIX"
] = r"C:\Users\eneemann\AppData\Local\ESRI\conda\envs\ml\tessdata"

# Start timer and print start time in UTC
start_time = time.time()
readable_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
print("The script start time is {}".format(readable_start))

# working_dir = r"C:\Users\eneemann\Desktop\Neemann\UDOT\Scans_for_Felix\S-R199(50)\conversion_tests"
working_dir = Path(
    r"C:\Users\eneemann\Desktop\Neemann\UDOT\Scans_for_Felix\S-R199(50)\conversion_tests"
)

rotate_dir = Path(
    r"C:\Users\eneemann\Desktop\Neemann\UDOT\Scans_for_Felix\S-R199(50)\rotate_tests"
)

# test_pdf_path = working_dir / "test.PDF"
test_pdf_path = working_dir / "multipage_test.pdf"
rotate_path = rotate_dir / "needs_rotated_180.jpg"

# Set Tesseract options
# These characters are not used: I, O,
wl = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ:-"
bl = ""
ocr_config = f"--oem 0 --psm 6 -c tessedit_char_whitelist={wl}"
# original options: ocr_config = "--oem 3 --psm 4"

# Build initial dataframe
# Could read in a working Excel file later
df = pd.DataFrame({"Number": [], "Filename": [], "Parcel": []})
df = df.astype({"Number": int, "Filename": object, "Parcel": object})
# df.dtypes

# out_csv = os.path.join(working_dir, "detected_parcels.csv")
out_csv = working_dir / "detected_parcels.csv"

# ocr_dir = os.path.join(working_dir, "images_to_ocr")
# if os.path.isdir(ocr_dir) == False:
#     os.mkdir(ocr_dir)

# os.chdir(ocr_dir)


def process(filename):
    if filename.suffix in [".pdf", ".PDF"]:
        convert_pymupdf(filename)


def convert(filename):
    basename = filename.stem
    pages = convert_from_path(filename, 300)
    num_pages = len(pages)
    print(f"number of pages: {num_pages}")
    if num_pages > 1:
        idx = 0
        for page in pages:
            idx += 1
            page.save(working_dir / f"{basename}_{idx}.jpg", "JPEG")
    else:
        pages[0].save(working_dir / f"{basename}.jpg", "JPEG")


def convert_pymupdf(filename):
    basename = filename.stem
    # mat = fitz.Matrix(300 / 72, 300 / 72)  # sets zoom factor for 300 dpi
    pages = fitz.open(filename)
    num_pages = len(pages)
    print(f"number of pages: {num_pages}")

    if num_pages > 1:
        idx = 0
        for page in pages:
            # page = pages.load_page(idx).get_pixmap(matrix=mat)
            page = pages.load_page(idx).get_pixmap(dpi=300)
            idx += 1
            page.save(working_dir / f"{basename}_mupdf_dpi_{idx}.jpg")

    else:
        # page = pages.load_page(0).get_pixmap(matrix=mat)
        page = pages.load_page(0).get_pixmap(dpi=300)
        page.save(working_dir / f"{basename}_mupdf_dpi.jpg")


def rotate(image_path):
    img = cv2.imread(str(image_path))
    h, w, b = img.shape  # shape provides height, width, bands
    print(f"Height: {h} \n Width: {w}")
    if h > w:
        print("the image will be rotated 90 degrees")
        rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(rotate_dir / "rotated_90cw_done.jpg"), rotated_90)
    else:
        print("the image will be rotated 180 degrees")
        rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(str(rotate_dir / "rotated_180_done.jpg"), rotated_180)


# process(test_pdf_path)

rotate(rotate_path)


#############################
# Skeleton processing chain #
#############################
# process(image)
# convert(image)
# rotate(image)
# prepare(image)
# detect(image)
# crop(image)
# prepare(image)
# detect(image)
# write(image)


###############################################
# Proof of concept script functions are below #
###############################################

# Iterate over images in the directory to detect circles and create cropped images to OCR
def process_directory_images(work_dir, ocr_directory):
    for item in os.listdir(work_dir):
        if item.casefold().endswith(("jpg", "tif", "tiff")):
            # item_path = os.path.join(work_dir, item)
            item_path = work_dir / item
            img = cv2.imread(item_path)
            # print(img.shape)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.blur(gray, (5, 5))
            # (thresh, img_bw) = cv2.threshold(gray_blur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Calculate circle radius
            # Get input size:
            dimensions = img.shape

            # height, width
            height = img.shape[0]
            width = img.shape[1]

            min_rad = math.ceil(0.01 * height) - 10
            max_rad = math.ceil(0.01 * height) + 10
            if min_rad < 15:
                min_rad = 15

            if max_rad < 30:
                max_rad = 30

            # print(min_rad, max_rad)
            inset = int(0.075 * max_rad)
            # print(inset)

            # Apply Hough transform on the blurred image.
            detected_circles = cv2.HoughCircles(
                gray_blur,
                cv2.HOUGH_GRADIENT,
                1,
                20,
                param1=50,
                param2=30,
                minRadius=min_rad,
                maxRadius=max_rad,
            )

            # Export images of detected circles
            if detected_circles is not None:
                export_circles_from_image(
                    detected_circles,
                    ocr_directory,
                    item_path,
                    img,
                    height,
                    width,
                    inset,
                )
            else:
                print(f"no circles found in {item_path}")


def export_circles_from_image(circles, ocr_d, file_path, im, hgt, wdth, ins):
    print(f"{len(circles[0])} circles found in {file_path}")
    # print(circles)

    # Convert the circle parameters a, b and r to integers.
    circles = np.uint16(np.around(circles))

    num = 0
    for i in circles[0, :]:
        # Prepare a black canvas on which to draw circles:
        canvas = np.zeros((hgt, wdth))
        # plt.imshow(canvas)

        # Draw a white circle on the canvas where detected:
        # color = (255, 255, 255)
        color = (255, 255, 255)
        thickness = -1
        centerX = i[0]
        centerY = i[1]
        radius = (
            i[2] - ins
        )  # inset the radius by 5 pixel to remove the circle and noise
        cv2.circle(canvas, (centerX, centerY), radius, color, thickness)
        # plt.imshow(canvas)

        # Create a copy of the input (3-band image) and mask input to white from the canvas:
        im_copy = im.copy()
        im_copy[canvas == 0] = (255, 255, 255)

        # Crop image to the roi:
        x = centerX - radius - 20
        y = centerY - radius - 20
        h = 2 * radius + 40
        w = 2 * radius + 40

        cropped_img = im_copy[y : y + h, x : x + w]
        # plt.figure()
        # plt.imshow(cropped_img)

        #         original_basename = original_filename.strip('.jpg')
        # original_basename = file_path.rsplit('.', 1)[0]
        # original_basename = os.path.basename(file_path).rsplit(".", 1)[0]
        original_basename = file_path.stem
        # out_file = os.path.join(ocr_d, str(f"{original_basename}_{num}.jpg"))
        out_file = ocr_d / f"{original_basename}_{num}.jpg"
        #         print(ocr_d)
        #         print(out_file)
        cv2.imwrite(out_file, cropped_img)

        num += 1


def ocr_images(directory, ocr_options, working_df):
    ocr_num = 0
    for ocr_file in os.listdir(directory):
        if ocr_file.endswith(".jpg"):
            ocr_count = ocr_file.split(".")[0].rsplit("_", 1)[1]

            print(f"working on: {ocr_file}")
            # print(ocr_count)
            # ocr_full_path = os.path.join(directory, ocr_file)
            ocr_full_path = directory / ocr_file
            ocr_img = cv2.imread(ocr_full_path)

            # Convert image to grayscale
            ocr_gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)
            # plt.figure()
            # plt.imshow(ocr_gray)

            # Median blur to reduce noise in the image
            # ocr_gray_blur = cv2.medianBlur(ocr_gray, 3)
            # plt.figure()
            # plt.imshow(ocr_gray_blur)

            ocr_text = pytesseract.image_to_string(ocr_gray, config=ocr_options)
            ocr_text = ocr_text.replace("\n", "").replace("\t", "").replace(" ", "")
            # new_row = {'Filename':ocr_file, 'Parcel':ocr_text}
            df_new = pd.DataFrame(
                {
                    "Number": [int(ocr_count)],
                    "Filename": [ocr_file],
                    "Parcel": [ocr_text],
                }
            )

            # ONCE YOU NOTICE THAT ALL THE PARCELS HAVE BEEN DETECTED ACCURTELY,
            # UNCOMMENT, TO APPEND
            # df = pd.concat([df, df_new])
            working_df = pd.concat([working_df, df_new], ignore_index=True, sort=False)
            # df = df.append(new_row, ignore_index=True)
            ocr_num += 1

    return working_df


# process_directory_images(working_dir, ocr_dir)
# final_df = ocr_images(ocr_dir, ocr_config, df)

# final_df.to_csv(out_csv)

print("Script shutting down ...")
# Stop timer and print end time in UTC
readable_end = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
print("The script end time is {}".format(readable_end))
print("Time elapsed: {:.2f}s".format(time.time() - start_time))
