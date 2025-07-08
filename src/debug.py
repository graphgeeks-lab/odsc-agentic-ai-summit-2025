"""
Debug script to compare a single record between the source and result FHIR data.
"""
import json
from pprint import pprint

INDEX_ID = 97

with open("../data/extracted_fhir.json", "r") as result, open("../data/fhir.json", "r") as source:
    source_data = json.load(source)
    result_data = json.load(result)

RESOURCE_TYPE = "Patient"

pprint([item["resource"] for item in source_data[INDEX_ID - 1]["entry"] if item["resource"]["resourceType"] == RESOURCE_TYPE])

pprint(result_data[INDEX_ID - 1])

