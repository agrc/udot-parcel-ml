#!/usr/bin/env python
# * coding: utf8 *
"""
server.py
a module that can receive web requests. this will be google cloud pub sub and
possibly github web hooks
"""

import logging
from os import getenv
from sys import stdout

from flask import Flask

import row_run

PORT = int(str(getenv("PORT"))) if getenv("PORT") else 8080
app = Flask(__name__)

logging.basicConfig(stream=stdout, level=logging.DEBUG)


@app.route("/", methods=["POST"])
def schedule():
    """schedule: the route that gcp cloud scheduler invokes"""
    row_run.ocr_all_mosaics()

    return ("", 204)


def start():
    """start: start the server for cloud run"""

    # This is used when running locally. Gunicorn is used to run the
    # application on Cloud Run. See entrypoint in Dockerfile.
    app.run(host="127.0.0.1", port=PORT, debug=False)


if __name__ == "__main__":
    start()
