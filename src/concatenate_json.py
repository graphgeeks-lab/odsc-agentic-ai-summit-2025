#!/usr/bin/env python3
"""
Because the BAML extract script is run in parallel batches, we get multiple JSON files.
Each file contains a JSON array of patient records. This script concatenates all JSON files in
the data/results directory into a single JSON file for the following purposes:
- Evaluation: We can evaluate the BAML extract script on the entire dataset.
- Graph building: We can build the graph on the entire dataset.

The script will also sort the records by `record_id` in ascending order to ensure that the right
records in the results are matched to the right source records.
"""

import glob
import json
import os
from pathlib import Path


def concatenate_json_files(input_dir: str, output_file: str) -> None:
    """
    Concatenate all JSON files in the input directory into a single JSON file.

    Args:
        input_dir: Directory containing JSON files to concatenate
        output_file: Path to the output concatenated JSON file
    """
    # Get all JSON files in the directory, sorted by name
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files to concatenate")

    # Combined array to hold all records
    all_records = []

    # Process each file
    for file_path in json_files:
        print(f"Processing {os.path.basename(file_path)}...")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ensure data is a list
            if isinstance(data, list):
                all_records.extend(data)
                print(f"  Added {len(data)} records (running total: {len(all_records)})")
            else:
                print(f"  Warning: {file_path} does not contain a JSON array (type: {type(data)})")

        except json.JSONDecodeError as e:
            print(f"  Error reading {file_path}: {e}")
        except Exception as e:
            print(f"  Unexpected error reading {file_path}: {e}")

    # Sort records by record_id in ascending order
    print(f"\nSorting {len(all_records)} records by record_id...")
    try:
        all_records.sort(key=lambda x: x.get("record_id", 0))
        print("Records sorted successfully")
    except Exception as e:
        print(f"Warning: Error sorting records: {e}")

    # Write the combined data to output file
    print(f"Writing {len(all_records)} total records to {output_file}...")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)

        print(f"Successfully created {output_file}")
        print(f"Total records: {len(all_records)}")
        print(f"\nExpected total records (27 files x 100): 2700")
        print(f"Actual total records: {len(all_records)}")

    except Exception as e:
        print(f"Error writing output file: {e}")


def main():
    """Main function to run the concatenation."""
    # Define paths
    input_directory = "../data/results"
    output_file = "../data/extracted_fhir.json"

    # Ensure input directory exists
    if not os.path.exists(input_directory):
        print(f"Input directory {input_directory} does not exist")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run concatenation
    concatenate_json_files(input_directory, output_file)


if __name__ == "__main__":
    main()
