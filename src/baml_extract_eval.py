"""
Script to evaluate the accuracy of the FHIR extraction results.

Results from BAML: ../data/results/extracted_fhir.json
Gold standard: ../data/fhir.json

The script assumes that the results from the gold standard are in the same order as the results from BAML.
"""

import argparse
import json
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

# Add a mapping of state abbreviations to full state names
STATE_ABBR_TO_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


@dataclass
class NameComparison:
    """Represents a name comparison between results and gold standard"""

    family_match: bool
    given_match: bool
    prefix_match: bool
    overall_match: bool
    record_id: int  # Store the record ID for failed matches


@dataclass
class AddressComparison:
    """Represents an address comparison between results and gold standard"""

    line_match: bool
    city_match: bool
    state_match: bool
    postal_code_match: bool
    country_match: bool
    overall_match: bool
    record_id: int  # Store the record ID for failed matches


@dataclass
class SimpleFieldComparison:
    """Represents simple field comparisons between results and gold standard"""

    gender_match: bool
    birth_date_match: bool
    marital_status_match: bool
    overall_match: bool
    record_id: int  # Store the record ID for failed matches


# --- Utility Functions ---
def get_first(arr) -> Any:
    if isinstance(arr, list) and arr:
        return arr[0]
    return None


def normalize_str(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, list):
        return get_first(val)
    return str(val)


def compare_strict(a, b) -> bool:
    return a == b


def compare_case_insensitive(a, b) -> bool:
    if a is None or b is None:
        return a == b
    return str(a).lower() == str(b).lower()


def compare_substring(a, b) -> bool:
    if a is None or b is None:
        return a == b
    a, b = str(a).lower(), str(b).lower()
    return a in b or b in a


def normalize_fhir_name(fhir_name_array) -> Dict[str, Any]:
    name = get_first(fhir_name_array) or {}
    return {
        "family": name.get("family"),
        "given": name.get("given", []),
        "prefix": get_first(name.get("prefix", [])),
    }


def normalize_fhir_address(fhir_address_array) -> Dict[str, Any]:
    address = get_first(fhir_address_array) or {}
    return {
        "line": get_first(address.get("line", [])),
        "city": address.get("city"),
        "state": address.get("state"),
        "postalCode": address.get("postalCode"),
        "country": address.get("country"),
    }


def normalize_fhir_marital_status(fhir_marital_status) -> Optional[str]:
    if not fhir_marital_status:
        return None
    text = fhir_marital_status.get("text")
    if text:
        return text.replace(" ", "")
    coding = fhir_marital_status.get("coding", [])
    if coding:
        return coding[0].get("display", "").replace(" ", "")
    return None


def combine_practitioner_name(practitioner_data: Dict[str, Any]) -> str:
    """Combine prefix, given, and family names from practitioner data"""
    if not practitioner_data or not isinstance(practitioner_data.get("name"), dict):
        return ""

    name = practitioner_data["name"]
    prefix = name.get("prefix", [])
    given = name.get("given", [])
    family = name.get("family")

    # Handle prefix as array (FHIR standard)
    if prefix and isinstance(prefix, list):
        prefix = get_first(prefix)

    # Handle given as array (FHIR standard)
    if given and isinstance(given, list):
        given = given  # Keep as list to extend later
    elif given and not isinstance(given, list):
        given = [given]
    else:
        given = []

    # Build the full name
    parts = []
    if prefix:
        parts.append(prefix)
    if given:
        parts.extend(given)
    if family:
        parts.append(family)

    return " ".join(parts).strip()


# --- Generalized Evaluation ---
def parse_fhir_bundle(fhir_json_str: str) -> Dict[str, Any]:
    if isinstance(fhir_json_str, dict):
        return fhir_json_str
    return json.loads(fhir_json_str)


def extract_patient_from_bundle(bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if bundle.get("resourceType") != "Bundle":
        return None
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            return resource
    return None


def extract_allergy_count_from_bundle(bundle: Dict[str, Any]) -> int:
    """Extract count of AllergyIntolerance resources from FHIR bundle"""
    if bundle.get("resourceType") != "Bundle":
        return 0

    count = 0
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "AllergyIntolerance":
            count += 1
    return count


def extract_immunization_count_from_bundle(bundle: Dict[str, Any]) -> int:
    """
    Extract count of Immunization resources from FHIR bundle, and also count Procedure resources
    mentioning vaccines/immunizations.
    """
    if bundle.get("resourceType") != "Bundle":
        return 0

    count = 0
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Immunization":
            count += 1
        elif resource.get("resourceType") == "Procedure":
            code = resource.get("code") or {}
            text = code.get("text", "") or ""
            if "vaccine" in text.lower() or "immunization" in text.lower():
                count += 1
                continue
            for coding in code.get("coding", []):
                display = coding.get("display", "") or ""
                if "vaccine" in display.lower() or "immunization" in display.lower():
                    count += 1
                    break
    return count


def extract_immunization_status_from_bundle(bundle: Dict[str, Any]) -> List[str]:
    """Extract status of all Immunization resources from FHIR bundle"""
    if bundle.get("resourceType") != "Bundle":
        return []

    statuses = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Immunization":
            status = resource.get("status")
            if status:
                statuses.append(status)
    return statuses


def extract_immunization_dates_from_bundle(bundle: Dict[str, Any]) -> List[str]:
    """Extract occurrenceDateTime of all Immunization resources from FHIR bundle"""
    if bundle.get("resourceType") != "Bundle":
        return []

    dates = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Immunization":
            # Check both occurrenceDateTime and occurrenceString
            date_time = resource.get("occurrenceDateTime")
            date_string = resource.get("occurrenceString")

            if date_time:
                dates.append(date_time)
            elif date_string:
                dates.append(date_string)
    return dates


def extract_all_practitioners_from_bundle(bundle: Dict[str, Any]) -> list:
    practitioners = set()
    if bundle.get("resourceType") != "Bundle":
        return list(practitioners)

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Encounter":
            participants = resource.get("participant", []) or []
            for participant in participants:
                individual = participant.get("individual", {})
                display = individual.get("display", "")
                if display:
                    practitioners.add(display)

    # Practitioner resources as before
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Practitioner":
            name_array = resource.get("name", [])
            if name_array:
                # Handle name as array (FHIR standard) - take the first name
                name = get_first(name_array)
                if name:
                    combined = combine_practitioner_name({"name": name})
                    if combined:
                        practitioners.add(combined)
    # print("DEBUG: Practitioners:", list(practitioners))
    return list(practitioners)


def read_n_lines(file_path: str, n: int) -> list:
    """Read up to n lines from a file and return as a list."""
    lines = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line)
    return lines


def evaluate_fields(
    eval_file_path: str, results_file_path: str, field_map: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, int], int, Dict[str, list], int, int]:
    """
    field_map: {
        field_name: {
            'extract_fhir': Callable[[Dict], Any],
            'extract_result': Callable[[Dict], Any],
            'compare': Callable[[Any, Any], bool]
        }
    }
    """
    stats = {k: 0 for k in field_map}
    total = 0
    failed_records = {k: [] for k in field_map}
    total_field_comparisons = 0
    total_failed_comparisons = 0
    with open(results_file_path, "r") as results_file:
        results_data_list = json.load(results_file)
    with open(eval_file_path, "r") as eval_file:
        eval_data_list = json.load(eval_file)
    # Truncate the eval_data_list to the same length as the results_data_list
    # This logic will break if the results are out of order, or if the results from BAML begin
    # at a record_id other than 1.
    eval_data_list = eval_data_list[: len(results_data_list)]
    for line_num, (eval_data, results_data) in enumerate(zip(eval_data_list, results_data_list), 1):
        fhir_bundle = parse_fhir_bundle(eval_data)
        patient = extract_patient_from_bundle(fhir_bundle)
        if not patient:
            continue
        for field, ops in field_map.items():
            total_field_comparisons += 1
            if (
                field.startswith("practitioner")
                or field.startswith("allergy")
                or field.startswith("immunization")
            ):
                # For practitioner, allergy, and immunization fields, use bundle data
                fhir_val = ops["extract_fhir"](fhir_bundle)
            else:
                # For patient fields, use patient data
                fhir_val = ops["extract_fhir"](patient)
            result_val = ops["extract_result"](results_data)
            if ops["compare"](result_val, fhir_val):
                stats[field] += 1
            else:
                failed_records[field].append(line_num)
                total_failed_comparisons += 1
        total += 1
    return stats, total, failed_records, total_field_comparisons, total_failed_comparisons


# --- Field Extractor Functions ---


def extract_family_fhir(patient: Dict[str, Any]) -> Any:
    return normalize_fhir_name(patient.get("name", []))["family"]


def extract_family_result(result: Dict[str, Any]) -> Any:
    return result.get("name", {}).get("family") if result.get("name") else None


def extract_given_fhir(patient: Dict[str, Any]) -> Any:
    return normalize_fhir_name(patient.get("name", []))["given"]


def extract_given_result(result: Dict[str, Any]) -> Any:
    return result.get("name", {}).get("given") if result.get("name") else None


def extract_prefix_fhir(patient: Dict[str, Any]) -> Any:
    return normalize_fhir_name(patient.get("name", []))["prefix"]


def extract_prefix_result(result: Dict[str, Any]) -> Any:
    return result.get("name", {}).get("prefix") if result.get("name") else None


def extract_line_fhir(patient: Dict[str, Any]) -> Any:
    return normalize_fhir_address(patient.get("address", []))["line"]


def extract_line_result(result: Dict[str, Any]) -> Any:
    return result.get("address", {}).get("line") if result.get("address") else None


def extract_city_fhir(patient: Dict[str, Any]) -> Any:
    return normalize_fhir_address(patient.get("address", []))["city"]


def extract_city_result(result: Dict[str, Any]) -> Any:
    return result.get("address", {}).get("city") if result.get("address") else None


def extract_state_fhir(patient: Dict[str, Any]) -> Any:
    return normalize_fhir_address(patient.get("address", []))["state"]


def extract_state_result(result: Dict[str, Any]) -> Any:
    state = result.get("address", {}).get("state") if result.get("address") else None
    if state is None:
        return None
    # Normalize to uppercase for lookup
    state_upper = str(state).strip().upper()
    return STATE_ABBR_TO_NAME.get(state_upper, state)


def extract_postalCode_fhir(patient: Dict[str, Any]) -> Any:
    return normalize_fhir_address(patient.get("address", []))["postalCode"]


def extract_postalCode_result(result: Dict[str, Any]) -> Any:
    return result.get("address", {}).get("postalCode") if result.get("address") else None


def extract_country_fhir(patient: Dict[str, Any]) -> Any:
    return normalize_fhir_address(patient.get("address", []))["country"]


def extract_country_result(result: Dict[str, Any]) -> Any:
    return result.get("address", {}).get("country") if result.get("address") else None


def extract_gender_fhir(patient: Dict[str, Any]) -> str | None:
    return patient["gender"].lower() if patient["gender"] else None


def extract_gender_result(result: Dict[str, Any]) -> str | None:
    return result["gender"].lower() if result["gender"] else None


def extract_birthDate_fhir(patient: Dict[str, Any]) -> Any:
    return patient.get("birthDate")


def extract_birthDate_result(result: Dict[str, Any]) -> Any:
    return result.get("birthDate")


def extract_maritalStatus_fhir(patient: Dict[str, Any]) -> Any:
    return normalize_fhir_marital_status(patient.get("maritalStatus"))


def extract_maritalStatus_result(result: Dict[str, Any]) -> Any:
    return result.get("maritalStatus")


def extract_practitioner_fhir(bundle: Dict[str, Any]) -> Any:
    return extract_all_practitioners_from_bundle(bundle)


def extract_practitioner_result(result: Dict[str, Any]) -> Any:
    return (
        combine_practitioner_name(result.get("practitioner", {}))
        if result.get("practitioner")
        else None
    )


def compare_practitioner(result: Any, fhir_list: Any) -> bool:
    return (
        result in fhir_list
        if fhir_list and result
        else (result is None or result == "")
        and (not fhir_list or all(not item for item in fhir_list))
    )


def extract_allergyRecordedCount_fhir(bundle: Dict[str, Any]) -> int:
    return extract_allergy_count_from_bundle(bundle)


def extract_allergyRecordedCount_result(result: Dict[str, Any]) -> int:
    allergy = result.get("allergy") or {}
    substance = allergy.get("substance") or []
    return len(substance) if substance else 0


def extract_immunizationCount_fhir(bundle: Dict[str, Any]) -> int:
    return extract_immunization_count_from_bundle(bundle)


def extract_immunizationCount_result(result: Dict[str, Any]) -> int:
    return len(result.get("immunization") or [])


def extract_immunizationDate_fhir(bundle: Dict[str, Any]) -> List[str]:
    return extract_immunization_dates_from_bundle(bundle)


def extract_immunizationDate_result(result: Dict[str, Any]) -> List[str]:
    return [
        imm.get("occurrenceDateTime") or imm.get("occurrenceString")
        for imm in (result.get("immunization") or [])
        if imm.get("occurrenceDateTime") or imm.get("occurrenceString")
    ]


def compare_immunizationDate(result: List[str], fhir_list: List[str]) -> bool:
    return (
        result[0] in fhir_list
        if result and fhir_list
        else (not result or not result[0])
        and (not fhir_list or all(not item for item in fhir_list))
    )


# --- Field Map Definitions ---
FIELD_MAP = {
    # Name fields
    "family": {
        "extract_fhir": extract_family_fhir,
        "extract_result": extract_family_result,
        "compare": compare_strict,
    },
    "given": {
        "extract_fhir": extract_given_fhir,
        "extract_result": extract_given_result,
        "compare": compare_strict,
    },
    "prefix": {
        "extract_fhir": extract_prefix_fhir,
        "extract_result": extract_prefix_result,
        "compare": compare_strict,
    },
    # Address fields
    "line": {
        "extract_fhir": extract_line_fhir,
        "extract_result": extract_line_result,
        "compare": compare_strict,
    },
    "city": {
        "extract_fhir": extract_city_fhir,
        "extract_result": extract_city_result,
        "compare": compare_strict,
    },
    "state": {
        "extract_fhir": extract_state_fhir,
        "extract_result": extract_state_result,
        "compare": compare_strict,
    },
    "postalCode": {
        "extract_fhir": extract_postalCode_fhir,
        "extract_result": extract_postalCode_result,
        "compare": compare_strict,
    },
    "country": {
        "extract_fhir": extract_country_fhir,
        "extract_result": extract_country_result,
        "compare": compare_strict,
    },
    # Simple fields
    "gender": {
        "extract_fhir": extract_gender_fhir,
        "extract_result": extract_gender_result,
        "compare": compare_case_insensitive,
    },
    "birthDate": {
        "extract_fhir": extract_birthDate_fhir,
        "extract_result": extract_birthDate_result,
        "compare": compare_strict,
    },
    "maritalStatus": {
        "extract_fhir": extract_maritalStatus_fhir,
        "extract_result": extract_maritalStatus_result,
        "compare": compare_strict,
    },
    # Practitioner fields
    "practitioner": {
        "extract_fhir": extract_practitioner_fhir,
        "extract_result": extract_practitioner_result,
        "compare": compare_practitioner,
    },
    # Allergy fields
    "allergyRecordedCount": {
        "extract_fhir": extract_allergyRecordedCount_fhir,
        "extract_result": extract_allergyRecordedCount_result,
        "compare": compare_strict,
    },
    # Immunization fields
    "immunizationCount": {
        "extract_fhir": extract_immunizationCount_fhir,
        "extract_result": extract_immunizationCount_result,
        "compare": compare_strict,
    },
    "immunizationDate": {
        "extract_fhir": extract_immunizationDate_fhir,
        "extract_result": extract_immunizationDate_result,
        "compare": compare_immunizationDate,
    },
}


# --- Reporting ---
def print_field_stats(
    stats: Dict[str, int],
    total: int,
    failed_records: Dict[str, list],
    total_field_comparisons: int,
    total_failed_comparisons: int,
) -> None:
    print("=== INFORMATION EXTRACTION EVALUATION RESULTS ===\n")
    print(f"Total Records: {total}")
    for field, count in stats.items():
        print(f"{field}: {count}/{total} ({count/total*100:.1f}%)")
        if failed_records[field]:
            # Print only the first 10 failed IDs for brevity
            print(f" First 10 failed IDs: {failed_records[field][:10]}")

    total_passed_comparisons = total_field_comparisons - total_failed_comparisons
    print(
        dedent(
            f"""
        Overall accuracy (across all fields):
        {total_passed_comparisons}/{total_field_comparisons} ({total_passed_comparisons/total_field_comparisons*100:.1f}%)
        """
        )
    )
    print(f"Total comparisons: {total_field_comparisons}")
    print(f"Failed comparisons: {total_failed_comparisons}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FHIR extraction results.")
    parser.add_argument(
        "--eval_file",
        "-e",
        type=str,
        default="../data/fhir.json",
        help="Path to the evaluation (gold standard) file.",
    )
    parser.add_argument(
        "--results_file",
        "-r",
        type=str,
        default="../data/extracted_fhir.json",
        help="Path to the results file.",
    )
    args = parser.parse_args()

    stats, total, failed_records, total_field_comparisons, total_failed_comparisons = (
        evaluate_fields(args.eval_file, args.results_file, FIELD_MAP)
    )
    print_field_stats(
        stats, total, failed_records, total_field_comparisons, total_failed_comparisons
    )
