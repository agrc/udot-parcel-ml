#!/usr/bin/env python
# * coding: utf8 *
"""
the file to run to start the project
"""

import logging
from pathlib import Path
from sys import stdout
from os import environ
from time import perf_counter
import google.cloud.logging
import google.cloud.storage
import pandas as pd

import row

logging.basicConfig(
    stream=stdout,
    format="%(levelname)-7s %(asctime)s %(module)10s:%(lineno)5s %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

start_seconds = perf_counter()

row.main()

# images = row.download("bucket-name", 0)

# for image in images:
#     print(image)
#     row.process(image)
#     row.convert(image)
#     row.rotate(image)
#     row.prepare(image)
#     row.detect(image)
#     row.crop(image)
#     row.prepare(image)
#     row.detect(image)
#     row.write(image)

logging.info("time taken for task %i: %s", TASK_INDEX, row.format_time(perf_counter() - start_seconds))
