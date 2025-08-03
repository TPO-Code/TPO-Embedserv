import click
import uvicorn
import requests  # We will use this for the 'stop' command
import json
from rich.console import Console
from rich.table import Table

from .models import pull_model, list_local_models, delete_model
from .manager import DEFAULT_KEEP_ALIVE_SECONDS
from .config import load_config, save_config

# We need a context object to pass the server URL around
class AppContext:
    def __init__(self):
        self.server_url = None


pass_ctx = click.make_pass_decorator(AppContext, ensure=True)

def get_default_server_url():
    """Reads config to build the default URL."""
    config = load_config()
    port = config.get('port', 11536) # Read from config, fall back to 11536
    return f"http://127.0.0.1:{port}"

@click.group()
@click.option(
    '--server-url',
    envvar='EMBEDSERV_URL',
    default=get_default_server_url(),
    help='The URL of the EmbedServ server.'
)
@pass_ctx

def main(ctx: AppContext, server_url: str):
    """
    EmbedServ: A local sentence transformer server.
    """
    ctx.server_url = server_url


# --- MODIFIED: The 'serve' command is now smarter ---
@main.command()
@click.option('--host', default='127.0.0.1', help='Host to bind the server to.')
@click.option(
    '--port',
    type=int,
    default=None,
    help='Port to bind to. Overrides the configured port.'
)
@click.option(
    '--keep-alive',
    default=None,
    type=click.IntRange(min=0),  # Use Click's built-in validation
    help='Default time (in seconds) to keep models loaded. Overrides configured value.'
)
def serve(host: str, port: int, keep_alive: int):
    """
    Starts the EmbedServ server.
    Reads config unless overridden by flags.
    """
    config = load_config()

    # Order of precedence: --flag > config file > hardcoded default
    port_to_use = port or config.get('port', 11536)

    # Check keep_alive flag; if not provided (is None), check config, then use default
    keep_alive_to_use = keep_alive
    if keep_alive_to_use is None:
        keep_alive_to_use = config.get('keep_alive', DEFAULT_KEEP_ALIVE_SECONDS)

    click.echo(f"🚀 Starting EmbedServ server on http://{host}:{port_to_use}")
    click.echo(f"   Default model keep-alive duration: {keep_alive_to_use} seconds")
    click.echo(f"   API documentation available at http://{host}:{port_to_use}/docs")

    from .server import app as fastapi_app
    fastapi_app.state.keep_alive_seconds = keep_alive_to_use
    uvicorn.run(fastapi_app, host=host, port=port_to_use, log_level="info")


@main.group()
def config():
    """View and manage persistent configuration."""
    pass


@config.command("set")
@click.argument("key")
@click.argument("value")
def set_config(key: str, value: str):
    """
    Set a configuration value. Supported keys: port, keep_alive.

    Example: embedserv config set port 1876
    Example: embedserv config set keep_alive 600
    """
    console = Console()
    key = key.lower()
    current_config = load_config()

    if key == 'port':
        try:
            port_val = int(value)
            if not (1024 <= port_val <= 65535):
                raise ValueError("Port must be between 1024 and 65535.")
            current_config[key] = port_val
        except ValueError as e:
            console.print(f"❌ [bold red]Invalid value for port:[/bold red] {e}")
            return
    elif key == 'keep_alive':
        try:
            keep_alive_val = int(value)
            if keep_alive_val < 0:
                raise ValueError("Keep-alive must be a non-negative integer.")
            current_config[key] = keep_alive_val
        except ValueError as e:
            console.print(f"❌ [bold red]Invalid value for keep_alive:[/bold red] {e}")
            return
    else:
        console.print(f"❌ [bold red]Error:[/bold red] '{key}' is not a supported configuration key.")
        console.print("   Supported keys are: [cyan]port[/cyan], [cyan]keep_alive[/cyan]")
        return

    console.print(f"Setting {key} to {value}...")
    save_config(current_config)
    console.print(f"✅ Configuration saved to ~/.embedserv/config.json")
    if key == 'port':
        console.print(
            f"[yellow]Restart the service for the new port to take effect.[/yellow]")
    else:
        console.print(
            f"[yellow]Restart the service for the new keep-alive default to take effect.[/yellow]")


@config.command("view")
def view_config():
    """Displays the current configuration."""
    console = Console()
    current_config = load_config()

    table = Table(title="EmbedServ Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")

    if not current_config:
        console.print("No configuration set. Using default values.")
        return

    for key, value in current_config.items():
        table.add_row(key, str(value))

    console.print(table)

@main.command("pull")
@click.argument("model_name")
def pull(model_name: str):
    """
    Downloads a model from Hugging Face.

    Example: embedserv pull all-MiniLM-L6-v2
    """
    pull_model(model_name)


@main.command("list")
def list_cmd():
    """Lists models available locally."""
    console = Console()
    models = list_local_models()

    table = Table(title="Local Models", show_header=True, header_style="bold magenta")
    table.add_column("Model Name", style="cyan", no_wrap=True)

    if not models:
        console.print("No local models found.")
        console.print("Use [bold]embedserv pull <model_name>[/bold] to download a model.")
        return

    for model_dir in models:
        table.add_row(model_dir)

    console.print(table)


@main.command("rm")
@click.argument("model_name") # Change argument name for clarity
def rm(model_name: str):
    """
    Deletes a local model by its Hugging Face name.
    """
    if click.confirm(f"Are you sure you want to delete the model '{model_name}'?"):
        delete_model(model_name)


@main.command("unload")
@pass_ctx
def unload(ctx: AppContext):
    """
    Stops the model running on the server by unloading it from memory.
    This does not shut down the server itself.
    """
    console = Console()
    console.print(f"Attempting to unload model on server at [cyan]{ctx.server_url}[/cyan]...")

    try:
        response = requests.post(f"{ctx.server_url}/api/v1/unload")
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()
        console.print(f"✅ Server response: [green]{data.get('message', 'Success!')}[/green]")

    except requests.exceptions.RequestException as e:
        console.print(f"❌ [bold red]Error connecting to server:[/bold red] {e}")
        console.print("   Please ensure the EmbedServ server is running.")
    except Exception as e:
        console.print(f"❌ [bold red]An unexpected error occurred:[/bold red] {e}")


# ---------------------------------
# --- New Database Command Group ---
# ---------------------------------

@main.group("db")
def db_group():
    """Manage vector database collections."""
    pass


@db_group.command("list")
@pass_ctx
def db_list(ctx: AppContext):
    """Lists all vector database collections on the server."""
    console = Console()
    try:
        response = requests.get(f"{ctx.server_url}/api/v1/db")
        response.raise_for_status()
        data = response.json()

        table = Table(title="Server Collections", show_header=True, header_style="bold green")
        table.add_column("Collection Name", style="yellow")

        collections = data.get('collections', [])
        if not collections:
            console.print("No collections found on the server.")
            return

        for name in collections:
            table.add_row(name)
        console.print(table)

    except requests.exceptions.RequestException as e:
        console.print(f"❌ [bold red]Error connecting to server:[/bold red] {e}")


@db_group.command("create")
@click.argument("name")
@pass_ctx
def db_create(ctx: AppContext, name: str):
    """Creates a new, empty collection on the server."""
    console = Console()
    try:
        response = requests.post(
            f"{ctx.server_url}/api/v1/db",
            json={"name": name}
        )
        response.raise_for_status()
        data = response.json()
        console.print(f"✅ Server response: [green]{data.get('message', 'Success!')}[/green]")
    except requests.exceptions.RequestException as e:
        console.print(f"❌ [bold red]Error communicating with server:[/bold red] {e}")
        if e.response:
            console.print(f"   [yellow]Details: {e.response.text}[/yellow]")


@db_group.command("delete")
@click.argument("name")
@pass_ctx
def db_delete(ctx: AppContext, name: str):
    """Deletes a collection and all its data from the server."""
    if not click.confirm(f"Are you sure you want to permanently delete the collection '{name}'?"):
        return

    console = Console()
    try:
        response = requests.delete(f"{ctx.server_url}/api/v1/db/{name}")
        response.raise_for_status()
        data = response.json()
        console.print(f"✅ Server response: [green]{data.get('message', 'Success!')}[/green]")
    except requests.exceptions.RequestException as e:
        console.print(f"❌ [bold red]Error communicating with server:[/bold red] {e}")
        if e.response:
            console.print(f"   [yellow]Details: {e.response.text}[/yellow]")

if __name__ == '__main__':
    main()