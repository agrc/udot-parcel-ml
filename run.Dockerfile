FROM gcr.io/buildpacks/gcp/run:v1
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
  libpoppler-dev poppler-utils tesseract-ocr libtesseract-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
USER
