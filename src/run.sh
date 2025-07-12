#!/bin/bash

source ../.venv/bin/activate

# Run the extract script for the specified range of record IDs
# uv run baml_extract.py --start 1 --end 100
# uv run baml_extract.py --start 101 --end 200
uv run baml_extract.py --start 201 --end 300
uv run baml_extract.py --start 301 --end 400
uv run baml_extract.py --start 401 --end 500
uv run baml_extract.py --start 501 --end 600
uv run baml_extract.py --start 601 --end 700
uv run baml_extract.py --start 701 --end 800
uv run baml_extract.py --start 801 --end 900
uv run baml_extract.py --start 901 --end 1000
uv run baml_extract.py --start 1001 --end 1100
uv run baml_extract.py --start 1101 --end 1200
uv run baml_extract.py --start 1201 --end 1300