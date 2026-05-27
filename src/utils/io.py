import json
from pathlib import Path


def load_text_file(path: str) -> str:
    """
    Read a text file securely using UTF-8 encoding.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_json_file(data, path: str) -> None:
    """
    Save any serializable Python object into a JSON file with pretty indent.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_file(path: str):
    """
    Load a JSON file into a Python object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
