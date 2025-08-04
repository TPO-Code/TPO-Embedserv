# EmbedServ 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**A local, Ollama-style server for sentence-transformer embeddings and semantic search.**

EmbedServ allows you to easily download and serve state-of-the-art sentence-transformer models via a REST API. It's designed for private, local-first use cases, with built-in memory management and an optional vector database for building powerful semantic search applications right on your machine.

It is built with a powerful stack of modern tools:
*   **[Sentence Transformers](https://www.sbert.net/)** for top-tier text embeddings.
*   **[FastAPI](https://fastapi.tiangolo.com/)** for a high-performance, self-documenting API.
*   **[ChromaDB](https://www.trychroma.com/)** for a simple, local-first vector store.
*   **[Click](https://click.palletsprojects.com/)** for a beautiful and intuitive command-line interface.

---

## ✨ Features

*   **Simple CLI**: Manage models and configuration with `pull`, `list`, `rm`, `config`, and `stop` commands.
*   **Powerful REST API**: Generate embeddings and manage vector databases programmatically. Interactive docs are available at `/docs`.
*   **Persistent Configuration**: Easily set the server port with `embedserv config set port <number>`.
*   **Centralized Caches**: Models and databases are stored centrally in `~/.embedserv/`.
*   **Automatic Memory Management**: The currently loaded model is automatically unloaded from memory after a configurable period of inactivity to save resources.
*   **Built-in Vector Database**: Create, delete, and manage multiple vector collections directly on the server.
*   **GPU Support**: Automatically uses your NVIDIA GPU if `torch` is installed with CUDA support.

---

## 🚀 Quick Install (Linux with systemd)

For Linux systems with `systemd`, using the automated installation script is the recommended method. It will set up EmbedServ to run as a background service that starts on boot.

Simply run the following command in your terminal:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/TPO-Code/TPO-Embedserv/master/install.sh)"
```

The script will guide you through the process. It installs the application and its dependencies into a self-contained directory at `~/embedserv`.

### Configuring the Port

By default, the server runs on port `11536`. To change this, use the `config` command:

```bash
# Set the server to use a new port
~/embedserv/venv/bin/embedserv config set port 8000

# Restart the service to apply the new port
sudo systemctl restart embedserv```

### Managing the Service

Once installed, you can manage the server with these commands:

- **Check status:** `sudo systemctl status embedserv`
- **View logs:** `sudo journalctl -u embedserv -f`
- **Stop the server:** `sudo systemctl stop embedserv`
- **Start the server:** `sudo systemctl start embedserv`

### Uninstall

To completely remove the application and the system service, run the uninstallation script:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/TPO-Code/TPO-Embedserv/master/uninstall.sh)"
```

---

## 🚀 Getting Started

If you performed the Quick Install, the server is already running. If you installed manually, start the server:

```bash
# If installed manually, start the server
embedserv serve
```
The server will start on the configured port (default: `http://127.0.0.1:11536`).

**1. Pull a Model**
In a new terminal, download a model. `all-MiniLM-L6-v2` is a great starting point.
```bash
embedserv pull all-MiniLM-L6-v2
```

**2. Generate Your First Embedding**
Use `curl` or any API client to interact with the server. This first call will load the model into memory.
```bash
curl -X POST http://127.0.0.1:11536/api/v1/embeddings \
-H "Content-Type: application/json" \
-d '{
    "model": "all-MiniLM-L6-v2",
    "input": "EmbedServ is awesome!"
}'
```

**3. Stop the Model**
To manually free up VRAM/RAM without waiting for the inactivity timer, unload the current model.
```bash
embedserv stop
```

---

## 🧠 Vector Database Quickstart

EmbedServ comes with a built-in vector database. Here's how to build a simple semantic search engine.

**Step 1: Load a model**
The server must have a model loaded to generate embeddings. The easiest way is to make an embedding request first.
```bash
# This command ensures all-MiniLM-L6-v2 is loaded
curl -X POST http://127.0.0.1:11536/api/v1/embeddings -H "Content-Type: application/json" -d '{"model": "all-MiniLM-L6-v2", "input": "loading model"}'
```

**Step 2: Create a collection**
```bash
embedserv db create my_document_collection
```

**Step 3: Add documents to the collection**
```bash
curl -X POST http://127.0.0.1:11536/api/v1/db/my_document_collection/add \
-H "Content-Type: application/json" \
-d '{
    "documents": [
        "The sky is a deep blue color today.",
        "Most types of grass are green.",
        "A car is a four-wheeled road vehicle that is powered by an engine."
    ],
    "metadatas": [{"source": "weather"},{"source": "botany"},{"source": "dictionary"}],
    "ids": ["doc1", "doc2", "doc3"]
}'
```

**Step 4: Query with natural language**
```bash
curl -X POST http://127.0.0.1:11536/api/v1/db/my_document_collection/query \
-H "Content-Type: application/json" \
-d '{
    "query_texts": ["what is an automobile?"],
    "n_results": 1
}'
```
The server will embed your query and find the most semantically similar document.

---

## 📚 CLI Command Reference

### Core Commands
*   `embedserv serve`: Starts the API server. Reads configuration from `~/.embedserv/config.json`.
    *   `--host <ip>`: Server host (default: `127.0.0.1`).
    *   `--port <int>`: Overrides the configured port for a single run.
    *   `--keep-alive <sec>`: Default time in seconds to keep models loaded (default: `300`).
*   `embedserv pull <model_name>`: Downloads a model from Hugging Face.
*   `embedserv list`: Lists all locally downloaded models.
*   `embedserv rm <model_dir_name>`: Deletes a local model.
*   `embedserv stop`: Unloads the currently active model on the server.

### Configuration Commands
*   `embedserv config set <key> <value>`: Sets a persistent configuration option.
    *   Example: `embedserv config set port 8000`
*   `embedserv config view`: Displays the current configuration.

### Database Commands
*   `embedserv db list`: Lists all database collections on the server.
*   `embedserv db create <name>`: Creates a new collection.
*   `embedserv db delete <name>`: Deletes a collection.

---

## 🛠️ Manual Installation & Development

If you are not on Linux or prefer not to use the automated script, you can install manually.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/TPO-Code/TPO-Embedserv.git
    cd TPO-Embedserv
    ```

2.  **Create a Virtual Environment**
    Using a virtual environment is strongly recommended to avoid dependency conflicts.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    Install the project in editable mode (`-e`) with its testing dependencies.
    ```bash
    pip install -e ".[test]"
    ```

4.  **Run the Server**
    ```bash
    embedserv serve
    ```

5.  **Run Tests**
    ```bash
    pytest
    ```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.