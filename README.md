# udot-parcel-ml

A repository for processing udot parcel images and extracting parcel numbers using machine learning.
Currently, this tool processes pdf's and images looking for circles. These circles are extracted and tiled and stored
to be run against the Google Cloud DocumentAI optical character recognition processor.

## Example source image

<img width="400" alt="image" src="https://user-images.githubusercontent.com/325813/217314819-c710e244-493d-4c3f-bc97-17bda56a0670.png">

## Example output

<img width="400" alt="image" src="https://user-images.githubusercontent.com/325813/217314532-8f376652-92b1-48d3-99b6-4359ee8ed74a.png">

This project is organized to work with build pack and Google Cloud Run Jobs and to run the commands locally via a CLI.

## CLI

To work with the CLI,

1. Create a python environment and install the `requirements.dev.txt` into that environment
1. Execute the CLI to see the commands and options available
   - `python row_cli.py`

## Workflow steps

1. generate an index of all files
1. filter the index to remove non image files and deeds

   `python row_cli.py index filter ./data/elephant/remaining_index.txt`

1. put the index in storage
1. run the job referencing the index location (edit the job name, file size, and task count)
1. generate another index from the resulting job

   `python row_cli.py storage generate-index --from=gs://ut-dts-agrc-udot-parcels-dev --prefix=elephant/mosaics/ --save-to=./data/elephant`

1. use a logging sink to add files with 0 circles detected and query for the file names and add that to the index generated in the previous step to avoid double processing files.
1. generate a remaining index between the original and the prior

   `python row_cli.py storage generate-remaining-index --full-index=gs://ut-dts-agrc-udot-parcels-dev --processed-index=./data/elephant --save-to=./data/elephant`

   _assuming the index in the bucket is the last remaining index for comparison_

1. filter out the deeds which have no circles

   `python row_cli.py index filter ./data/elephant/remaining_index.txt`

1. move the current index into the job and replace with the remaining index renamed as index.txt
1. repeat 4-9 until there are no more files left to process
1. Authentication for document ai job

   - activate your terminal as a service account

     `gcloud auth activate-service-account email@address --key-file=/path/to/sa.json`

1. start the job

   `python row_cli.py process circles --job=elephant --from=gs://bucket--save-to=bucket --index=gs://bucket --task-index=0 --file-count=1 --instances=1 --project=1234 --processor=123abc`

## Postprocessing, filtering results, generating final products

1. organize the OCR results into one cloud storage location
   - place all parquet (.gz) files in one "folder" (ex: prefix = alligator)
1. download and combine the OCR results to local file storage (./data)
   
   `python row_cli.py ocr-results download alligator --from=gs://ut-dts-agrc-udot-parcels-dev --save-to=./data`

1. clean the OCR results, save output locally (./data/cleaned)

   `python row_cli.py ocr-results clean ./data/ocr_results/combined_ocr_results.gz --save-to=./data/cleaned`

1. join the UDOT spreadsheet info to the OCR results, save locally (./data/joined)

   `python row_cli.py ocr-results join ./data/cleaned/cleaned-ocr-results-2023-03-28-09-28.csv --save-to=./data/joined`

1. filter the joined results to produced the final products, save locally (./data/filtered)

   `python row_cli.py ocr-results filter ./data/joined/joined-ocr-results-2023-03-28-09-31.csv --save-to=./data/filtered`

   - 3 CSV files will be saved with the following naming conventions:
      1. "final-good-ocr-results-{%YY-%mm-%dd-%HH-%MM}.csv"
      1. "final-bad-ocr-results-{%YY-%mm-%dd-%HH-%MM}.csv"
      1. "final-all-ocr-results-{%YY-%mm-%dd-%HH-%MM}.csv"
