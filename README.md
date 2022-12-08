# udot-parcel-ml

A repository for processing udot parcel images and extracting parcel numbers using machine learning

This project is organized to work with build pack and Google Cloud Run Jobs and to run the commands locally via a CLI.

## CLI

To work with the CLI,

1. Create a python environment and install the `requirements.dev.txt` into that environment
1. Execute the CLI to see the commands and options available
   - `python row_cli.py`


### Examples

Get files while testing

```py
python row_cli.py storage get_job_files --from-bucket=test/data --task-index=0 --testing=true
```

Get files from bucket

```py
python row_cli.py storage get_job_files --from-bucket=ut-udot-row-county-parcels --task-index=0
```
