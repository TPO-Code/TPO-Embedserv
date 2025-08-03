import shutil
import re

from rich.console import Console
from sentence_transformers import SentenceTransformer

from .config import MODELS_DIR, ensure_dirs_exist


def _hf_name_to_cache_dir(model_name: str) -> str:
    """Converts a Hugging Face model name to its cache directory name."""
    # Example: 'sentence-transformers/all-MiniLM-L6-v2' -> 'models--sentence-transformers--all-MiniLM-L6-v2'
    # The 'models--' prefix is added by sentence-transformers when caching.
    sanitized_name = model_name.replace("/", "--")
    return f"models--{sanitized_name}"


def _cache_dir_to_hf_name(cache_dir: str) -> str:
    """Converts a cache directory name back to a Hugging Face model name."""
    if not cache_dir.startswith("models--"):
        return cache_dir

    name_part = cache_dir[len("models--"):]
    # Check if a '--' separator exists, indicating an org/repo structure.
    if '--' in name_part:
        # Replace only the first occurrence, which separates org and repo.
        return name_part.replace('--', '/', 1)
    else:
        # No org, so the name is just the rest of the string.
        return name_part


def pull_model(model_name: str):
    """
    Downloads a sentence-transformer model from Hugging Face, showing a status spinner.
    """
    ensure_dirs_exist()
    console = Console()
    console.print(f"📦 Pulling model [bold cyan]{model_name}[/bold cyan]...")
    console.print(f"   Models are stored in: {MODELS_DIR}")

    # Use rich.status to show a spinner during the blocking download operation.
    try:
        with console.status(f"[bold green]Downloading...[/bold green]", spinner="earth"):
            # This blocking call will now execute while the spinner is visible.
            SentenceTransformer(model_name, cache_folder=str(MODELS_DIR))

        # This will only print if the download was successful.
        console.print(f"✅ Successfully pulled model [bold green]{model_name}[/bold green].")

    except Exception as e:
        # The 'with' context manager will automatically stop the spinner on an exception.
        console.print(f"\n❌ [bold red]Error pulling model:[/bold red] {e}")
        console.print("   Please check the model name and your internet connection.")


def list_local_models() -> list[str]:
    """
    Lists the models that have been downloaded locally by traversing the
    huggingface_hub cache structure.
    """
    ensure_dirs_exist()
    models = []
    if not MODELS_DIR.exists():
        return []

    # A model is stored in a directory like 'models--sentence-transformers--all-MiniLM-L6-v2'
    for repo_dir in MODELS_DIR.iterdir():
        if not repo_dir.is_dir():
            continue

        # The actual model files are in a 'snapshots' subdirectory.
        snapshots_path = repo_dir / "snapshots"
        if not snapshots_path.exists() or not snapshots_path.is_dir():
            continue

        # Inside 'snapshots' are directories named after commit hashes.
        for commit_dir in snapshots_path.iterdir():
            # A valid sentence-transformer model will have 'modules.json'.
            if commit_dir.is_dir() and (commit_dir / "modules.json").exists():
                # We found a valid model. The 'name' is the top-level repo directory.
                hf_name = _cache_dir_to_hf_name(repo_dir.name)
                models.append(hf_name)
                # Break the inner loop; no need to check other snapshots for this repo.
                break

    return sorted(models)


def delete_model(model_name: str) -> bool:
    """
    Deletes a model from local storage using its Hugging Face name.
    """
    ensure_dirs_exist()
    console = Console()

    model_dir_name = _hf_name_to_cache_dir(model_name)  # Convert to cache dir name
    model_path = MODELS_DIR / model_dir_name

    if not model_path.exists() or not model_path.is_dir():
        console.print(f"❌ [bold red]Error:[/bold red] Model not found: {model_name}")
        console.print("   Run [bold]embedserv list[/bold] to see available models.")
        return False

    try:
        console.print(f"🗑️  Deleting model [bold cyan]{model_name}[/bold cyan]...")
        shutil.rmtree(model_path)
        console.print(f"✅ Successfully deleted model.")
        return True
    except OSError as e:
        console.print(f"❌ [bold red]Error deleting model directory:[/bold red] {e}")
        return False