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
1. run the job referencing the index location
1. generate another index from the resulting job
   `python row_cli.py storage generate-index --from=gs://ut-dts-agrc-udot-parcels-dev --prefix=elephant/mosaics/ --save-to=./data/elephant`
1. generate a remaining index between the original and the prior
1. repeat 3-7
