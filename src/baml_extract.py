"""
This script runs the BAML pipeline to extract information from the FHIR unstructured patient notes data
and outputs the results to newline-delimited JSON files.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
from dotenv import load_dotenv

os.environ["BAML_LOG"] = "WARN"

from baml_client import b

load_dotenv()


async def extract_patient(record: Dict[str, str]) -> Dict[str, Any]:
    patient = b.ExtractPatient(record["note"])
    output = patient.model_dump()
    # Clean up output
    output["record_id"] = record["record_id"]
    output["maritalStatus"] = output["maritalStatus"].value if output["maritalStatus"] else None
    print(f"Extracted patient details for {record['record_id']}")
    return output


async def extract_practitioner(record: Dict[str, str]) -> Dict[str, Any]:
    practitioner = b.ExtractPractitioner(record["note"])
    output = practitioner.model_dump()
    print(f"Extracted practitioner details for {record['record_id']}")
    return output


async def extract_immunization(record: Dict[str, str]) -> List[Dict[str, Any]]:
    immunization = b.ExtractImmunization(record["note"])
    output = [i.model_dump() for i in immunization]
    print(f"Extracted immunization for {record['record_id']}")
    return output


async def extract_allergy(record: Dict[str, str]) -> Dict[str, Any]:
    allergy = b.ExtractAllergy(record["note"])
    output = allergy.model_dump()
    print(f"Extracted allergy for {record['record_id']}")
    return output


async def process_record(record: Dict[str, str]) -> Dict[str, Any]:
    # Patient
    patient_result = await extract_patient(record)
    # Practitioner
    practitioner_result = await extract_practitioner(record)
    if not all(v is None for v in practitioner_result.values()):
        patient_result["practitioner"] = practitioner_result
    # Immunization
    immunization_result = await extract_immunization(record)
    if not all(v is None for v in immunization_result):
        patient_result["immunization"] = immunization_result
    # Allergy
    allergy_result = await extract_allergy(record)
    if allergy_result.get("substance"):
        patient_result["allergy"] = allergy_result
    return patient_result


async def extract(records: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    tasks = [process_record(record) for record in records]
    return await asyncio.gather(*tasks)


async def main(fname: str, start: int, end: int) -> None:
    "Run the information extraction workflow"
    df = pl.read_json(fname)
    records = df.to_dicts()
    records = records[start - 1 : end]

    results = await extract(records)
    # Output the results to a newline-delimited JSON file
    Path("../data/results").mkdir(parents=True, exist_ok=True)
    results_df = pl.DataFrame(results)
    results_df.write_json(f"../data/results/extracted_fhir_{args.start}_{args.end}.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="Start index")
    parser.add_argument("--end", type=int, default=10, help="End index")
    parser.add_argument("--fname", type=str, default="../data/note.json", help="Input file name")
    parser.add_argument(
        "--output",
        type=str,
        default="../data/results/extracted_fhir.json",
        help="Output file name",
    )
    args = parser.parse_args()
    if args.start < 1:
        raise ValueError("Start index must be 1 or greater")

    asyncio.run(main(args.fname, start=args.start, end=args.end))
