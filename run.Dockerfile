FROM gcr.io/buildpacks/gcp/run:v1
USER root
RUN useradd -s /bin/bash dummy
RUN apt-get update && apt-get install -y --no-install-recommends \
  libpoppler-dev poppler-utils tesseract-ocr libtesseract-dev libgl1-mesa-glx && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
