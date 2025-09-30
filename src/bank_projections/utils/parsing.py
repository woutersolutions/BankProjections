import datetime
from collections.abc import Iterable
from typing import Any


def read_date(value: str | datetime.date | datetime.datetime) -> datetime.date:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    raise ValueError(f"Cannot convert {value} to date")


def read_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ["true", "yes", "1"]:
            return True
        elif value in ["false", "no", "0"]:
            return False
    raise ValueError(f"Cannot convert {value} to bool")


def get_identifier(input_identifier: str, reference_identifiers: Iterable[str]) -> str:
    cleaned = strip_identifier(input_identifier)
    for id in reference_identifiers:
        if cleaned == strip_identifier(id):
            return id
    raise KeyError(f"{input_identifier} not found in identifiers")


def get_identifiers(input_identifiers: Iterable[str], reference_identifiers: list[str]) -> list[str]:
    return [get_identifier(identifier, reference_identifiers) for identifier in input_identifiers]


def correct_identifier_keys(input_dict: dict[str, Any], reference_identifiers: list[str]) -> dict[str, Any]:
    return dict(zip(get_identifiers(input_dict.keys(), reference_identifiers), input_dict.values(), strict=True))


def is_in_identifiers(identifier: str, identifiers: Iterable[str]) -> bool:
    return strip_identifier(identifier) in [strip_identifier(id2) for id2 in identifiers]


def strip_identifier_keys(input_dict: dict[str, Any]) -> dict[str, Any]:
    return {strip_identifier(key): value for key, value in input_dict.items()}


def strip_identifier(identifier: str | None) -> str | None:
    if identifier is None:
        return None
    else:
        return (
            identifier.strip()
            .lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("/", "")
            .replace("\\", "")
        )
