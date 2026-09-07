import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.json"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"config.json not found at {CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


_config = load_config()

BACKEND_HOST = _config["backend"]["host"]
BACKEND_PORT = _config["backend"]["port"]
BACKEND_URL = _config["backend"]["base_url"]