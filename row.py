#!/usr/bin/env python
# * coding: utf8 *
"""
UDOT Right of Way (ROW) Parcel Number Extraction
Right of way module containing methods
"""
import logging
import math
import re
import unicodedata
from datetime import datetime
from io import BytesIO
from itertools import islice
from os import environ
from pathlib import Path
from time import perf_counter

import cv2
import google.cloud.documentai
import google.cloud.logging
import google.cloud.storage
import numpy as np
import pandas as pd
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError, InvalidArgument, RetryError
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from PIL.Image import DecompressionBombError

if "PY_ENV" in environ and environ["PY_ENV"] == "production":
    LOGGING_CLIENT = google.cloud.logging.Client()
    STORAGE_CLIENT = google.cloud.storage.Client()

    LOGGING_CLIENT.setup_logging()

TASK_RESULTS = []


def mosaic_all_circles(job_name, input_bucket, output_location, file_index, task_index, task_count, total_size):
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
    logging.info("job name: %s task: %i processing %s files", job_name, task_index, files)

    #: Initialize GCP storage client and bucket
    bucket = STORAGE_CLIENT.bucket(input_bucket[5:])

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
            logging.info(
                "job name: %s task: %i conversion time %s: %s",
                job_name,
                task_index,
                format_time(perf_counter() - conversion_start),
                {"file": object_name, "pages": count, "message": messages},
            )

        elif extension in [".jpg", ".jpeg", ".tif", ".tiff", ".png"]:
            images = list([bucket.blob(object_name).download_as_bytes()])
        else:
            logging.info('job name: %s task: %i not a valid document or image: "%s"', job_name, task_index, object_name)

            continue

        #: Process images to get detected circles
        logging.info("job name: %s task: %i detecting circles in %s", job_name, task_index, object_name)
        all_detected_circles = []
        circle_start = perf_counter()

        for image in images:
            circle_images = get_circles_from_image_bytes(image, None, object_name)
            all_detected_circles.extend(circle_images)  #: extend because circle_images will be a list

        logging.info(
            "job name: %s task: %i circle detection time taken %s: %s",
            job_name,
            task_index,
            object_name,
            format_time(perf_counter() - circle_start),
        )

        circle_count = len(all_detected_circles)
        if circle_count == 0:
            logging.warning("job name: %s task: %i 0 circles detected in %s", job_name, task_index, object_name)

        #: Process detected circle images into a mosaic
        logging.info("job name: %s task: %i mosaicking images in %s", job_name, task_index, object_name)
        mosaic_start = perf_counter()

        mosaic = build_mosaic_image(all_detected_circles, object_name, None)

        logging.info(
            "job name: %s task: %i image mosaic time taken %s: %s",
            job_name,
            task_index,
            object_name,
            format_time(perf_counter() - mosaic_start),
        )

        logging.info(
            "job name: %s task: %i total time taken for entire task %s",
            job_name,
            task_index,
            format_time(perf_counter() - object_start),
        )

        upload_mosaic(mosaic, output_location, object_name, job_name)


def ocr_all_mosaics(inputs):
    """the code to run in the cloud run job

    Args:
        inputs (class): the inputs to the function
            job_name (str): the name of the run job. typically named after an animal in alphabetical order
            input_bucket (str): the bucket to get files from using the format `gs://bucket-name`
            output_location (str): the location to save the results to. omit the `gs://` prefix
            quota (int): the number of files to process
            file_index (str): the path to the folder containing an `index.txt` file listing all the images in a bucket.
                            `gs://bucket-name`
            project_number (int): the number of the gcp project
            processor_id (str): the id of the documentai processor

    Returns:
        A list of lists with the results of the OCR
    """
    #: Get files to process for this job
    index = get_index(inputs.file_index)

    if index is None:
        logging.error("job name: %s failed to get index", inputs.job_name)

        return

    files_to_process = []
    failed_to_process = []

    i = 0
    with index.open(mode="r", encoding="utf-8", newline=None) as data:
        while i < inputs.quota:
            line = data.readline()

            if line:
                files_to_process.append(line)
                i += 1
            else:
                break

    if len(files_to_process) == 0:
        logging.warning("job is completed")

        return

    logging.info("job name: %s processing %i files %s", inputs.job_name, len(files_to_process), files_to_process)

    #: Initialize GCP storage client and bucket
    bucket = STORAGE_CLIENT.bucket(inputs.input_bucket[5:])

    options = ClientOptions(api_endpoint="us-documentai.googleapis.com")
    ai_client = google.cloud.documentai.DocumentProcessorServiceClient(client_options=options)

    processor_name = ai_client.processor_path(inputs.project_number, "us", inputs.processor_id)

    #: Iterate over objects to detect circles and perform OCR
    for object_name in files_to_process:
        object_start = perf_counter()
        object_name = object_name.rstrip()

        if not Path(object_name).name.casefold().endswith(("jpg", "jpeg", "tif", "tiff", "png")):
            logging.info("job name: %s item is incorrect file type: %s", inputs.job_name, object_name)

            continue

        image_content = []
        try:
            image_content = bucket.blob(object_name).download_as_bytes()
        except Exception as ex:
            logging.warning("job name: %s failed to download %s: %s", inputs.job_name, object_name, ex)

            continue

        logging.info(
            "job name: %s download finished %s: %s",
            inputs.job_name,
            format_time(perf_counter() - object_start),
            {"file": object_name},
        )

        raw_document = google.cloud.documentai.RawDocument(content=image_content, mime_type="image/jpeg")
        request = google.cloud.documentai.ProcessRequest(name=processor_name, raw_document=raw_document)

        result = None
        try:
            result = ai_client.process_document(request=request)
            logging.info(
                "job name: %s ocr finished %s: %s",
                inputs.job_name,
                format_time(perf_counter() - object_start),
                {"file": object_name},
            )
        except (RetryError, InternalServerError) as error:
            logging.warning(
                "job name: %s ocr failed on %s. %s",
                inputs.job_name,
                object_name,
                error.message,
            )

            failed_to_process.append(object_name + "\n")

            continue
        except InvalidArgument as error:
            logging.warning(
                "job name: %s ocr failed on %s. %s\n%s",
                inputs.job_name,
                object_name,
                error.message,
                error.details,
            )

            failed_to_process.append(object_name + "\n")

            continue

        TASK_RESULTS.append([object_name, result.document.text])

    upload_results(
        TASK_RESULTS, inputs.output_location, f"ocr-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}", inputs.job_name
    )

    i = 0
    with index.open(mode="r+", encoding="utf-8", newline=None) as data:
        while i < inputs.quota:
            data.readline()
            i += 1

        remaining_files = data.read()
        data.seek(0)
        data.write(remaining_files)
        data.writelines(failed_to_process)
        data.truncate()

    update_index(bucket, index)

    return TASK_RESULTS


def update_index(bucket, path_object):
    """renames the current index and uploads the index file to the bucket

    Args:
        bucket (Bucket): the bucket to upload the index file to
        path_object (Path): the index file to upload
    """
    bucket.rename_blob(bucket.blob("index.txt"), f"index-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt")

    with path_object.open(mode="r+b", newline=None) as data:
        bucket.blob("index.txt").upload_from_file(data, content_type="text/plain")


def generate_index(from_location, prefix, save_location):
    """reads file names from the `from_location` and optionally saves the list to the `save_location` as an index.txt
    file. Prefix can optionally be included to narrow down index location. Cloud storage buckets must start with `gs://`
    Args:
        from_location (str): the directory to read the files from. Prefix GSC buckets with gs://.
        prefix (str): subdirectory or GCS prefix. This prefix will also be stripped from the beginning of GCS paths.
        save_location (str): the directory to save the list of files to. An index.txt file will be created within this
                             directory
    Returns:
        list(str): a list of file names
    """

    files = list([])

    logging.info('reading files from "%s"', from_location)
    if from_location.startswith("gs://"):
        iterator = STORAGE_CLIENT.list_blobs(from_location[5:], max_results=None, versions=False, prefix=prefix)

        files = [blob.name.removeprefix(prefix).strip() for blob in iterator]
    else:
        from_location = Path(from_location)

        if prefix:
            from_location = from_location / prefix

        if not from_location.exists():
            logging.warning("from location %s does not exists", from_location)

            return files

        iterator = from_location.glob("**/*")

        files = [str(item).strip() for item in iterator if item.is_file()]

    if save_location is None:
        return files

    if save_location.startswith("gs://"):
        bucket = STORAGE_CLIENT.bucket(save_location[5:])
        blob = bucket.blob("index.txt")

        with BytesIO() as data:
            for item in files:
                data.write(str(item).encode("utf-8") + b"\n")

            blob.upload_from_string(data.getvalue())
    else:
        save_location = Path(save_location)

        if not save_location.exists():
            save_location.mkdir(parents=True, exist_ok=True)

        with save_location.joinpath("index.txt").open("w", encoding="utf-8", newline=None) as output:
            for item in files:
                output.write(str(item) + "\n")

    return files


def download_file_from(bucket_name, file_name):
    """downloads `file_name` from `bucket_name`. Index path object is returned.
    Cloud storage buckets must start with `gs://`
    Args:
        bucket_name (str): the bucket where the file resides. Prefix GSC buckets with gs://.
        file_name (number): the file name (with extension, i.e. index.txt) to download
    Returns:
        index (Path): path object for the downloaded file
    """
    index = Path(__file__).parent / ".ephemeral" / file_name

    if not bucket_name.startswith("gs://"):
        logging.warning("bucket name %s does not start with gs://", bucket_name)

        return None

    bucket = STORAGE_CLIENT.bucket(bucket_name[5:])

    blob = bucket.blob(file_name)

    if not index.parent.exists():
        index.parent.mkdir(parents=True)

    try:
        blob.download_to_filename(str(index))
    except Exception as ex:
        logging.error("error downloading file index %s. %s", index, ex, exc_info=True)

        raise ex

    return index


def get_index(from_location):
    """generic function to get index from cloud storage or local directory. Index path object is returned.
    Cloud storage buckets must start with `gs://`
    Args:
        from_location (str): the bucket or local directory where the index.txt file resides.
                             Prefix GSC buckets with gs://.
    Returns:
        index (Path): path object for the index.txt file
    """

    if from_location.startswith("gs://"):
        index = download_file_from(from_location, "index.txt")
    else:
        folder = Path(from_location)

        if not folder.exists():
            raise FileNotFoundError("folder does not exist")

        index = folder.joinpath("index.txt")

        if not index.exists():
            raise FileNotFoundError("index.txt file does not exist")

    return index


def get_first_and_last_index(task_index, task_count, total_size):
    """calculates a range of indexes based on the task index and total number of tasks. This is used to split up the
    index file
    Args:
        task_index (number): the index of the current cloud run task
        task_count (number): the total number of cloud run tasks
        total_size (number): the total number of files to process

    Returns:
        tuple(number, number): the first index and last index
    """
    job_size = math.ceil(total_size / task_count)
    first_index = task_index * job_size
    last_index = task_index * job_size + job_size

    return first_index, last_index


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
    index = get_index(from_location)

    if index is None:
        return []

    task_index = int(task_index)
    task_count = int(task_count)
    total_size = int(total_size)

    first_index, last_index = get_first_and_last_index(task_index, task_count, total_size)

    file_list = []

    with index.open("r", encoding="utf-8", newline=None) as data:
        file_list = list(islice(data, first_index, last_index))

    logging.info("task number %i will work on file indices from %i to %i", task_index, first_index, last_index)

    return file_list


def generate_remaining_index(full_index_location, processed_index_location, save_location):
    """reads file names from the `from_location` and optionally saves the list to the `save_location` as an index.txt
    file. Cloud storage buckets must start with `gs://`
    Args:
        full_index_location (str): the location from which to read the full index. Prefix GSC buckets with gs://.
        processed_index_location (str): the location from which to read the already-processed index.
                                        Prefix GSC buckets with gs://.
        save_location (str): the directory to save the list of files to. An index.txt file will be created within this
                             directory
    Returns:
        list(str): a list of file names
    """

    #: Get all files from the full index
    full_index = get_index(full_index_location)

    if full_index is None:
        return []

    with full_index.open(mode="r", encoding="utf-8", newline=None) as data:
        all_files = {l.strip() for l in data.readlines()}

    logging.info("total number of files %i", len(all_files))

    #: Get already-processed files from processed index
    processed_index = get_index(processed_index_location)

    if processed_index is None:
        return []

    with processed_index.open(mode="r", encoding="utf-8", newline=None) as data:
        processed_files = {l.strip() for l in data.readlines()}

    logging.info("number of already-processed files %i", len(processed_files))

    #: Get the difference to determine what remaining files need to be processed
    remaining_files = all_files - processed_files
    logging.info("number of remaining files to process %i", len(remaining_files))

    if save_location is None:
        return remaining_files

    if save_location.startswith("gs://"):
        bucket = STORAGE_CLIENT.bucket(save_location[5:])
        blob = bucket.blob("remaining_index.txt")

        with BytesIO() as data:
            for item in remaining_files:
                data.write(str(item).encode("utf-8") + b"\n")

            blob.upload_from_string(data.getvalue())
    else:
        save_location = Path(save_location)

        if not save_location.exists():
            logging.warning("save location %s does not exists", save_location)

            return remaining_files

        with save_location.joinpath("remaining_index.txt").open("w", encoding="utf-8", newline=None) as output:
            for item in remaining_files:
                output.write(str(item) + "\n")

    return remaining_files


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

    images = (convert_to_bytes(image) for image in images if image is not None)

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
    img = None
    try:
        img = cv2.imdecode(np.frombuffer(byte_img, dtype=np.uint8), 1)  # 1 means flags=cv2.IMREAD_COLOR
    except Exception as ex:
        logging.error("unable to read image from bytes: %s, %s", file_name, ex)

        return []

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

        min_rad = max(math.ceil(ratio_multiplier * height) - fudge_value, 15)
        max_rad = max(math.ceil(ratio_multiplier * height) + fudge_value, 30)

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
        logging.info("no circles detected for %s", file_name)

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
        crop_x = min(max(center_x - radius - 20, 0), width)
        crop_y = min(max(center_y - radius - 20, 0), height)
        crop_height = 2 * radius + 40
        crop_width = 2 * radius + 40

        masked_image = image_copy[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]

        if out_dir:
            original_basename = Path(file_name).stem
            out_file = out_dir / f"{original_basename}_{i}.jpg"
            cv2.imwrite(str(out_file), masked_image)

        masked_images.append(masked_image)

    return masked_images


def upload_results(data, bucket_name, out_name, job_name):
    """upload results dataframe to a GCP bucket as a gzip file

    Args:
        data (list): a list containing the results for the task (a list of lists. the first index being the file name
                     and the second being the text found)
        bucket_name (str): the name of the destination bucket
        out_name (str): the name of the gzip file

    Returns:
        nothing
    """
    file_name = f"{job_name}/{out_name}.gz"
    logging.info("uploading %s to %s/%s", out_name, bucket_name, file_name)

    bucket = STORAGE_CLIENT.bucket(bucket_name)
    new_blob = bucket.blob(file_name)

    frame = pd.DataFrame(data, columns=["file_name", "text"])

    with BytesIO() as parquet:
        frame.to_parquet(parquet, compression="gzip")

        new_blob.upload_from_string(parquet.getvalue(), content_type="application/gzip")


def format_time(seconds):
    """seconds: number
    returns a human-friendly string describing the amount of time
    """
    minute = 60.00
    hour = 60.00 * minute

    if seconds < 30:
        return f"{int(seconds * 1000)} ms"

    if seconds < 90:
        return f"{round(seconds, 2)} seconds"

    if seconds < 90 * minute:
        return f"{round(seconds / minute, 2)} minutes"

    return f"{round(seconds / hour, 2)} hours"


def download_run(bucket, run_name):
    """download a runs worth of results from a GCP bucket

    Args:
        bucket (str): the name of the bucket
        run_name (str): the name of the run

    Returns:
        str: the location of the files
    """
    bucket = STORAGE_CLIENT.bucket(bucket)
    blobs = bucket.list_blobs(prefix=run_name)
    location = Path(__file__).parent / "data"

    if not location.joinpath(run_name).exists():
        location.joinpath(run_name).mkdir(parents=True)

    for blob in blobs:
        if blob.name.endswith(".gz"):
            blob.download_to_filename(location / blob.name)

    return location.joinpath(run_name)


def download_ocr_results(bucket_name, run_name, out_dir):
    """download ocr results from a GCP bucket

    Args:
        bucket (str): the name of the bucket
        run_name (str): the name of the run to get files from
        out_dir (str): where to save the results

    Returns:
        str: the location of the files
    """
    if bucket_name.startswith("gs://"):
        bucket_name = bucket_name[5:]

    bucket = STORAGE_CLIENT.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=run_name)
    location = Path(__file__).parent / "data"

    ocr_dir = location.joinpath(f"ocr_results/{run_name}")
    ocr_dir.mkdir(parents=True, exist_ok=True)

    #: download .gz files
    logging.info("downloading .gz files from cloud storage")
    [blob.download_to_filename(ocr_dir / blob.name) for blob in blobs if blob.name.endswith(".gz")]

    ocr_files = ocr_dir.glob("*.gz")

    logging.info("combining %i files into a single dataframe", len(ocr_files))

    #: combine results into a single dataframe
    dfs = []
    for ocr_file in ocr_files:
        temp_df = pd.read_parquet(ocr_file)
        dfs.append(temp_df)

    combined_df = pd.concat(dfs)
    orig_length = len(combined_df.index)
    logging.info("rumber of rows before de-duplicating: %i", orig_length)
    combined_df.drop_duplicates(inplace=True, ignore_index=True)

    final_length = len(combined_df.index)
    diff = orig_length - final_length
    logging.info("rumber of rows after removing duplicates: %i", final_length)
    logging.info("removed %i duplicate rows", diff)

    out_file = Path(out_dir) / "ocr_results" / "combined_ocr_results.gz"
    combined_df.to_parquet(out_file, compression="gzip")
    logging.info("saved combined ocr results to %s", out_file)

    #: delete downloaded files
    logging.info("deleting individual ocr files")
    [Path(ocr_file).unlink() for ocr_file in ocr_files]

    return out_dir


def clean_ocr_results(original_results_file, out_dir):
    """clean ocr results down to quality results by cleaning up the initial results

    Args:
        original_results_file (str): path to the parquet file with original combined results (path_to_file.gz)
        out_dir (str): where to save the CSV file results

    Returns:
        str: the location of the output CSV file with cleaned results
    """
    #: silence pandas SettingWithCopyWarning
    pd.options.mode.chained_assignment = None

    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_parquet(original_results_file)

    orig_length = len(results_df.index)
    logging.info("rumber of rows before cleanup: %i", orig_length)

    #: Add column for the original UDOT filename
    results_df["udot_file_name"] = results_df.apply(lambda r: r["file_name"].split("/mosaics/", 1)[1].strip(), axis=1)

    #: Remove spaces and newline characters adjacent to colons
    results_df["text"] = results_df.apply(lambda r: r["text"].replace(":\n", ":").strip(), axis=1)
    results_df["text"] = results_df.apply(lambda r: r["text"].replace("\n:", ":").strip(), axis=1)
    results_df["text"] = results_df.apply(lambda r: r["text"].replace(": ", ":").strip(), axis=1)
    results_df["text"] = results_df.apply(lambda r: r["text"].replace(" :", ":").strip(), axis=1)
    #: Then remove newline characters and replace with spaces
    results_df["text"] = results_df.apply(lambda r: r["text"].replace("\n", " ").strip(), axis=1)

    #: Convert string to list
    results_df["text"] = results_df.apply(lambda r: r["text"].split(), axis=1)

    #: Remove alpha-only items - not relevant, should contain a number
    results_df["text"] = results_df.apply(lambda r: [item for item in r["text"] if not item.isalpha()], axis=1)

    #: Remove rows where length of text list is zero
    results_df = results_df[results_df["text"].apply(lambda r: len(r)) > 0]

    #: Convert list column to string
    results_df["text"] = results_df.apply(lambda r: " ".join(r["text"]), axis=1)

    #: Check number of rows before/after removing duplicates
    intermediate_length = len(results_df.index)
    logging.info("rumber of rows before de-duplicating: %i", intermediate_length)
    results_df.drop_duplicates(inplace=True, ignore_index=True)

    final_length = len(results_df.index)
    diff = intermediate_length - final_length
    logging.info("rumber of rows after removing duplicates: %i", final_length)
    logging.info("removed %i duplicate rows", diff)

    #: Save output locally
    out_file = out_dir / f"cleaned-ocr-results-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv"
    results_df.to_csv(out_file)
    logging.info("saved cleaned ocr results to %s", out_file)

    return out_dir


def join_spreadsheet_info(cleaned_results_file, out_dir):
    """join additional information from the udot spreadsheets to the cleaned ocr results

    Args:
        cleaned_results_file (str): path to the csv file with cleaned ocr results (path_to_file.csv)
        out_dir (str): where to save the CSV file with cleaned and joined results

    Returns:
        str: the location of the output CSV file with cleaned and joined results
    """
    #: silence pandas SettingWithCopyWarning
    pd.options.mode.chained_assignment = None

    #: set up variables
    spreadsheet_dir = Path(__file__).parent / "spreadsheets"
    out_dir = Path(out_dir)

    #: combine spreadsheets into a single df for joining
    dfs = []
    for sheet in spreadsheet_dir.glob("*.xlsx"):
        prefix = sheet.stem
        temp_df = pd.read_excel(sheet)
        #: add udot_file_name field for joining later
        temp_df["udot_file_name"] = temp_df.apply(lambda r: f"{prefix}/{r['name']}", axis=1)
        dfs.append(temp_df)

    combined_df = pd.concat(dfs)

    #: join the spreadsheet info to the results
    results_df = pd.read_csv(cleaned_results_file)
    joined_df = pd.merge(results_df, combined_df, left_on="udot_file_name", right_on="udot_file_name", how="inner")

    #: calculate the URLs (file in udot projectwise, file in udot cloud storage, mosaic in ugrc cloud storage)
    joined_df["projectwise_url"] = joined_df.apply(
        lambda r: f"https://connect-projectwisewac.bentley.com/pwlink?datasource=Bentley.PW--udot-pw.bentley.com~3Audot-pw-02&objectId={r['guid']}&objectType=doc&app=pwe",
        axis=1,
    )
    joined_df["udot_url"] = joined_df.apply(
        lambda r: f"https://storage.cloud.google.com/ut-udot-row-county-parcels/{r['udot_file_name']}", axis=1
    )
    joined_df["mosaic_url"] = joined_df.apply(
        lambda r: f"https://storage.cloud.google.com/ut-dts-ugrc-udot-parcel-mosaics/{r['udot_file_name']}", axis=1
    )

    #: compare length before and after removing duplicates
    initial_length = len(joined_df.index)
    logging.info("rumber of rows before de-duplicating: %i", initial_length)
    joined_df.drop_duplicates(["udot_file_name"], inplace=True, ignore_index=True)

    final_length = len(joined_df.index)
    diff = initial_length - final_length
    logging.info("rumber of rows after removing duplicates: %i", final_length)
    logging.info("removed %i duplicate rows", diff)

    #: parse text column into multiple rows (this is how UDOT wants it)
    exploded_df = joined_df.copy()
    new_col = exploded_df["text"].str.split(" ").apply(pd.Series, 1).stack()
    new_col.index = new_col.index.droplevel(-1)  # to line up with main df's index
    new_col.name = "text"  # needs a name to join
    del exploded_df["text"]  # remove the original column
    final_exploded_df = exploded_df.join(new_col)
    logging.info("rumber of rows after exploding ocr results: %i", len(final_exploded_df.index))

    #: save results to CSV
    out_file = out_dir / f"joined-ocr-results-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv"
    final_exploded_df.to_csv(out_file)
    logging.info("saved cleaned and joined ocr results to %s", out_file)

    return out_dir


def _filter_too_many_letters(df):
    df["too_many_letters"] = "pass"
    mask = (df["text"].str.contains("^[^:18]*$")) & (df["text"].str.contains("(?:[A-Z][^A-Z]*){4,}"))
    df.loc[mask, "keep"] = "no"
    df.loc[mask, "too_many_letters"] = "fail"
    logging.info("Number of parcels with 4+ letters and no colon, 1, or 8 flagged: %i", mask.value_counts()[1])

    return df


def _filter_five_number_run(df):
    df["five_number_run"] = "pass"
    mask = (df["text"].str.contains("^[^:18]*$")) & (df["text"].str.contains(r"\d{5,}"))
    df.loc[mask, "keep"] = "no"
    df.loc[mask, "five_number_run"] = "fail"
    logging.info("Number of parcels with 5+ numbers and no colon, 1, or 8 flagged: %i", mask.value_counts()[1])

    return df


def filter_results(previous_results_file, out_dir):
    """filter ocr results down further by filtering out additional irrelevant patterns

    Args:
        previous_results_file (str): path to the CSV file with cleaned and joined results (path_to_file.csv)
        out_dir (str): where to save the CSV file results

    Returns:
        str: the location of the output CSV file with filtered results
    """

    #: silence pandas SettingWithCopyWarning
    pd.options.mode.chained_assignment = None

    #: set up variables
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    working_df = pd.read_csv(previous_results_file)

    #: ensure the 'text' field is read as a string
    working_df["text"] = working_df["text"].astype(str)

    #: add a 'keep' field, initialized as 'yes', but be used to mark rows to remove with 'no'
    working_df["keep"] = "yes"

    #: upper-case all letters
    working_df["text"] = working_df["text"].str.upper()

    #: remove periods, parenthesis, double colons, and slashes
    logging.info("Removing periods, parenthesis, double colons, and slashes")
    empty = ""
    working_df["text"] = working_df.apply(
        lambda r: r["text"]
        .replace(".", empty)
        .replace("(", empty)
        .replace(")", empty)
        .replace("/", empty)
        .replace("\\", empty)
        .replace("::", empty)
        .strip(),
        axis=1,
    )

    #: remove leading colons
    logging.info("Removing leading colons")
    working_df["text"] = working_df["text"].apply(lambda x: x.lstrip(":"))

    #: remove trailing exclamation points
    working_df["text"] = working_df["text"].apply(lambda x: x.rstrip("!"))

    #: replace euro symbol with 'E'
    logging.info("Replacing euro symbol with 'E'")
    working_df["text"] = working_df.apply(lambda r: r["text"].replace("€", "E").strip(), axis=1)

    #: replaced accented characters with non-accented characters
    logging.info("Replacing accented characters with non-accented characters")
    working_df["text"] = working_df.apply(
        lambda r: "".join(c for c in unicodedata.normalize("NFD", r["text"]) if unicodedata.category(c) != "Mn"), axis=1
    )

    #: flag parcels that contains special characters (other than a colon)
    #: 123:2A = pass
    #: 10-CI = fail
    #: #605" = fail
    working_df["special_char"] = "pass"
    mask = working_df["text"].str.contains(r"[^\w:]") == True
    working_df.loc[mask, "keep"] = "no"
    working_df.loc[mask, "special_char"] = "fail"
    logging.info("Number parcels with special characters flagged: %i", mask.value_counts()[1])

    #: flag parcels that start with a letter or non-digit, should start with a number
    #: 123:2A = pass
    #: A23:2A = fail
    #: TYP:S7 = fail
    working_df["nondigit_start"] = "pass"
    mask = ~working_df["text"].str.contains(r"^\d") == True
    working_df.loc[mask, "keep"] = "no"
    working_df.loc[mask, "nondigit_start"] = "fail"
    logging.info("Number parcels starting with a non-digit flagged: %i", mask.value_counts()[1])

    #: flag parcels with a ':P' pattern; permits, not relevant - but PUE is okay!
    #: 123:PUE = pass
    #: 123:P = fail
    #: 123:PR = fail
    working_df["permit"] = "pass"
    mask = working_df["text"].str.contains(r"(?!:PUE):P") == True
    working_df.loc[mask, "keep"] = "no"
    working_df.loc[mask, "permit"] = "fail"
    logging.info("Number of permits (:P pattern, but not :PUE) flagged: %i", mask.value_counts()[1])

    #: flag parcels with a colon, that do not have a number before the colon
    #: most of these also fail the nondigit_start filter
    #: 123:2A = pass
    #: ABC:2A = fail
    working_df["no_number_before_colon"] = "pass"
    mask = working_df["text"].str.contains(r"^\D*:") == True
    working_df.loc[mask, "keep"] = "no"
    working_df.loc[mask, "no_number_before_colon"] = "fail"
    logging.info("Number of colons not preceeded by a number flagged: %i", mask.value_counts()[1])

    #: flag parcels longer than 13 characters
    #: 1649NT:STPUEQ = pass
    #: 1649NTS:STPUEQR = fail
    working_df["too_long"] = "pass"
    mask = working_df["text"].str.len() > 13
    working_df.loc[mask, "keep"] = "no"
    working_df.loc[mask, "too_long"] = "fail"
    logging.info("Number of parcels exceeding 13 characters flagged: %i", mask.value_counts()[1])

    #: flag parcels with 4 or more letters given that no colon exists (or 1 or 8)
    #: the colon is often confused as a 1 or 8, so if those exist, the parcel is kept (UDOT prefers this)
    #: 1572NT:2EC = pass (has a colon)
    #: 3:ST2EQ = pass (has a colon)
    #: 31ST2EQ = pass (1 might be a confused colon)
    #: 3ST2EQ = fail (4 letters and no colon, 1, or 8)
    #: 4BARS = fail
    #: 25DBIA = fail

    working_df = _filter_too_many_letters(working_df)

    #: remove parcels with 5 or more numbers in a row (and not a colon, 1, or a 8)
    #: the colon is often confused as a 1 or 8, so if those exist, the parcel is kept (UDOT prefers this)
    #: 1572NT:2EC = pass
    #: 23582A = pass (8 might be a confused colon)
    #: 2572:2A = pass (has a colon)
    #: 257212A = pass (1 might be a confused colon)
    #: 03702M = fail
    #: 699062ON = fail

    working_df = _filter_five_number_run(working_df)

    #: remove parcels where text = '0'
    working_df["zero"] = "pass"
    mask = working_df["text"] == "0"
    working_df.loc[mask, "keep"] = "no"
    working_df.loc[mask, "zero"] = "fail"
    logging.info("Number of parcels equal to 0 flagged: %i", mask.value_counts()[1])

    #: drop duplicates on the 'udot_file_name' and 'text' fields
    before_length = len(working_df.index)
    logging.info("rumber of rows before final de-duplication: %i", before_length)
    working_no_duplicates = working_df.drop_duplicates(["udot_file_name", "text"], inplace=False, ignore_index=True)

    after_length = len(working_no_duplicates.index)
    duplicate_diff = before_length - after_length
    logging.info("rumber of rows after removing duplicates: %i", after_length)
    logging.info("removed %i duplicate rows", duplicate_diff)

    #: only keep desired column names
    keep_cols = [
        "file_name",
        "udot_file_name",
        "name",
        "project_number",
        "project_name",
        "guid",
        "projectwise_url",
        "udot_url",
        "mosaic_url",
        "text",
        "keep",
        "special_char",
        "nondigit_start",
        "permit",
        "no_number_before_colon",
        "too_long",
        "too_many_letters",
        "five_number_run",
        "zero",
    ]

    working_no_duplicates = working_no_duplicates[keep_cols]

    #: save all results
    #: save results to CSV
    out_file_all = out_dir / f"final-all-ocr-results-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv"
    working_no_duplicates.to_csv(out_file_all)
    logging.info("saved final all ocr results to %s", out_file_all)

    #: save only good results to CSV
    keeps = working_no_duplicates[working_no_duplicates["keep"] == "yes"]
    out_file_keeps = out_dir / f"final-good-ocr-results-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv"
    keeps.to_csv(out_file_keeps)
    logging.info("saved final all ocr results to %s", out_file_keeps)

    #: save only bad results to CSV
    discards = working_no_duplicates[working_no_duplicates["keep"] == "no"]
    out_file_discards = out_dir / f"final-bad-ocr-results-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv"
    discards.to_csv(out_file_discards)
    logging.info("saved final all ocr results to %s", out_file_discards)

    pre_dup = working_df[working_df["keep"] == "yes"]
    dup_diff = len(pre_dup.index) - len(keeps.index)
    logging.info("good duplicates removed: %i", dup_diff)

    return out_dir


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
        logging.info("no images to mosaic for %s", object_name)

        return np.array(None)

    object_path = Path(object_name)
    max_dim = 0
    buffer = 5

    #: Loop through all images to get dimensions, save largest dimension
    all_widths = []
    all_heights = []
    for img in images:
        all_widths.append(img.shape[1])
        all_heights.append(img.shape[0])

    max_dim = max([max(all_widths), max(all_heights)])

    #: Set up parameters for mosaic, calculate number of cols/rows
    number_images = len(images)
    number_columns = math.floor(math.sqrt(number_images))
    number_rows = math.ceil(number_images / number_columns)

    #: Build mosaic image with white background
    tile_width = max_dim + 2 * buffer
    total_height = tile_width * number_rows
    total_width = tile_width * number_columns

    logging.info(
        "mosaicking %i images into %i column by %i row grid, %s",
        number_images,
        number_columns,
        number_rows,
        {"square pixels": tile_width, "file name": object_name},
    )

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


def upload_mosaic(image, bucket_name, object_name, job_name):
    """upload mosaic image to a GCP bucket as a jpeg mime type

    Args:
        image (np.array): the mosaic image bytes
        bucket_name (str): the name of the destination bucket
        object_name (str): the name of the image object (original filename)

    Returns:
        bool: True if successful, False otherwise
    """
    #: Upload mosaic image to GCP bucket
    if image is None or not image.any():
        logging.info('no mosaic image created or uploaded: "%s"', object_name)

        return False

    file_name = f"{job_name}/mosaics/{object_name}"
    logging.info("uploading %s to %s/%s", object_name, bucket_name, file_name)

    bucket = STORAGE_CLIENT.bucket(bucket_name)
    new_blob = bucket.blob(str(file_name))

    is_success, buffer = cv2.imencode(".jpg", image)

    if not is_success:
        logging.error("unable to encode image: %s", object_name)

    new_blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")

    return True
