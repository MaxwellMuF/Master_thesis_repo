from pathlib import Path
from typing import Any, Dict
import json


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file and ensure it contains a dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file, but found: {path}")

    with path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    if not isinstance(json_data, dict):
        raise ValueError("Expected a dict at the top level of the JSON.")

    return json_data


def save_json(json_data: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary as JSON to a file. Ensures the folder exists.
    """
    if not isinstance(json_data, dict):
        raise ValueError("Expected a dict at the top level of the JSON.")

    if not path.parent.exists():
        raise FileNotFoundError(f"Parent folder does not exist: {path.parent}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
