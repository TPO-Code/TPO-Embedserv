import os
import json
from pathlib import Path

# --- Application-wide Configuration ---

def get_app_dir() -> Path:
    """
    Returns the application's root directory.
    Defaults to ~/.embedserv, but can be overridden by the EMBEDSERV_HOME environment variable.
    """
    if env_home := os.environ.get("EMBEDSERV_HOME"):
        return Path(env_home)
    return Path.home() / ".embedserv"


APP_DIR = get_app_dir()
MODELS_DIR = APP_DIR / "models"
DB_DIR = APP_DIR / "databases"
CONFIG_FILE = APP_DIR / "config.json"

def ensure_dirs_exist():
    """
    Ensures that the application's data directories exist.
    """
    APP_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    DB_DIR.mkdir(exist_ok=True)

def load_config() -> dict:
    """Loads the persistent configuration from the JSON file."""
    ensure_dirs_exist()
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_config(config: dict):
    """Saves the configuration dictionary to the JSON file."""
    ensure_dirs_exist()
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)