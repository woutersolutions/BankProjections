import datetime
from typing import Iterable


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


def get_identifier(identifier: str, identifiers: Iterable[str]) -> str:
    cleaned = clean_identifier(identifier)
    for id in identifiers:
        if cleaned == clean_identifier(id):
            return id
    raise KeyError(f"{identifier} not found in identifiers")


def is_in_identifiers(identifier: str, identifiers: Iterable[str]) -> bool:
    return clean_identifier(identifier) in [clean_identifier(id) for id in identifiers]


def clean_identifier(identifier: str | None) -> str | None:
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
