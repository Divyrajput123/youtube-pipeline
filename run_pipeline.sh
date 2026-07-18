#!/bin/bash
# Run the pipeline with output to both terminal and log file
cd /Users/divysingh/Downloads/template-python-main
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
python -u -m pipeline --config config.json 2>&1
