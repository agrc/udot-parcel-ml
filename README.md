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
1. filter the index (if necessary)
1. put the index in storage
1. run the job referencing the index location (edit the job name, file size, and task count)
1. generate another index from the resulting job
   `python row_cli.py storage generate-index --from=gs://ut-dts-agrc-udot-parcels-dev --prefix=elephant/mosaics/ --save-to=./data/elephant`
1. use a logging sink to add files with 0 circles detected and query for the file names and add that to the index generated in the previous step to avoid double processing files.
1. generate a remaining index between the original and the prior
   `python row_cli.py storage generate-remaining-index --full-index=gs://ut-dts-agrc-udot-parcels-dev --processed-index=./data/elephant --save-to=./data/elephant`
   _assuming the index in the bucket is the last remaining index for comparison_
1. move the current index into the job and replace with the remaining index renamed as index.txt
1. repeat 4-8 until there are no more files left to process
