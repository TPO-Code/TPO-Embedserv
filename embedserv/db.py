# In embedserv/db.py

import logging
from typing import Optional

import chromadb
from chromadb.types import Collection
from sentence_transformers import SentenceTransformer

from .config import DB_DIR, ensure_dirs_exist

log = logging.getLogger(__name__)


class VectorDBManager:
    """
    Manages vector database collections using ChromaDB.
    """

    def __init__(self):
        ensure_dirs_exist()
        # Initialize a persistent ChromaDB client that saves to our app directory
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        log.info(f"VectorDBManager initialized. Database path: {DB_DIR}")

    def list_collections(self) -> list[str]:
        """Lists all available collection names."""
        return [c.name for c in self.client.list_collections()]

    def create_collection(self, name: str):
        """
        Creates a new collection, raising a ValueError if it already exists.
        """
        # --- MODIFICATION START ---
        # First, check if a collection with this name is already present.
        if name in self.list_collections():
            # Raise a specific, identifiable ValueError.
            error_msg = f"Collection '{name}' already exists. Please delete it first if you want to recreate it."
            log.warning(error_msg)
            raise ValueError(error_msg)
        # --- MODIFICATION END ---

        self.client.create_collection(name=name)
        log.info(f"Created collection '{name}'.")

    def delete_collection(self, name: str):
        """Deletes a collection."""
        self.client.delete_collection(name=name)
        log.info(f"Deleted collection '{name}'.")

    def count_documents_in_collection(self, name: str) -> int:
        """Counts the number of documents in a collection."""
        collection = self.get_collection(name)
        count = collection.count()
        log.info(f"Collection '{name}' contains {count} documents.")
        return count

    def get_collection(self, name: str) -> Collection:
        """
        Retrieves a collection, raising a ValueError if it doesn't exist.
        """
        try:
            return self.client.get_collection(name=name)
        except ValueError as e:
            log.error(f"Collection '{name}' not found.")
            raise ValueError(f"Collection '{name}' does not exist.") from e

    def add_to_collection(
            self,
            collection_name: str,
            documents: list[str],
            metadatas: list[dict],
            ids: list[str],
            model: SentenceTransformer
    ):
        """
        Embeds documents using the provided model and adds them to a collection.
        """
        collection = self.get_collection(collection_name)
        log.info(f"Generating embeddings for {len(documents)} documents to add to '{collection_name}'...")
        embeddings = model.encode(documents, convert_to_tensor=True)

        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        log.info(f"Successfully added {len(documents)} documents to '{collection_name}'.")

    def query_collection(
            self,
            collection_name: str,
            query_texts: list[str],
            n_results: int,
            model: SentenceTransformer,
            where: Optional[dict] = None
    ) -> dict:
        """
        Embeds query texts and performs a similarity search on a collection.
        """
        collection = self.get_collection(collection_name)
        log.info(f"Querying collection '{collection_name}' with {len(query_texts)} text(s)...")
        if where: # Add logging for visibility
            log.info(f"Applying metadata filter: {where}")

        query_embeddings = model.encode(query_texts, convert_to_tensor=True)

        results = collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=n_results,
            where=where
        )
        return results

    def get_from_collection(self, collection_name: str, ids: list[str], include: list[str]):
        """Retrieves documents and their metadata by ID."""
        collection = self.get_collection(collection_name)
        log.info(f"Getting {len(ids)} documents by ID from '{collection_name}'.")
        return collection.get(ids=ids, include=include)

    def update_in_collection(
            self,
            collection_name: str,
            ids: list[str],
            model: SentenceTransformer,
            documents: list[str] = None,
            metadatas: list[dict] = None
    ):
        """Updates documents and/or metadatas for given IDs."""
        collection = self.get_collection(collection_name)
        log.info(f"Updating {len(ids)} documents by ID in '{collection_name}'.")

        update_kwargs = {}

        if metadatas is not None:
            update_kwargs['metadatas'] = metadatas

        if documents is not None:
            # If documents are being updated, we MUST re-calculate their embeddings.
            log.info(f"Re-calculating embeddings for {len(documents)} updated document(s)...")
            new_embeddings = model.encode(documents, convert_to_tensor=True).tolist()
            update_kwargs['documents'] = documents
            update_kwargs['embeddings'] = new_embeddings

        if not update_kwargs:
            log.warning("Update called with no documents or metadatas to update.")
            return # Nothing to do

        collection.update(
            ids=ids,
            **update_kwargs
        )

    def delete_from_collection(self, collection_name: str, ids: list[str]):
        """Deletes documents from a collection by ID."""
        collection = self.get_collection(collection_name)
        log.info(f"Deleting {len(ids)} documents by ID from '{collection_name}'.")
        return collection.delete(ids=ids)