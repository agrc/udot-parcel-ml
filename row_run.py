#!/usr/bin/env python
# * coding: utf8 *
"""
the file to run to start the project
"""
from time import perf_counter
import row

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
