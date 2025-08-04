# In embedserv/db.py

import logging
from typing import Optional, List, Iterator, Dict, Any

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

    def list_collections_with_metadata(self) -> List[Dict[str, Any]]:
        """Returns a list of dictionaries, each containing a collection's name and metadata."""
        collections = self.client.list_collections()
        return [{"name": c.name, "metadata": c.metadata} for c in collections]

    def create_collection(self, name: str, model_name: str):
        """
        Creates a new collection, embedding the model name in its metadata.
        """
        if name in self.list_collections():
            error_msg = f"Collection '{name}' already exists..."
            raise ValueError(error_msg)

        # Store the model name in the collection's permanent metadata
        collection_metadata = {"embedding_model": model_name}
        self.client.create_collection(name=name, metadata=collection_metadata)
        log.info(f"Created collection '{name}' and associated it with model '{model_name}'.")

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

    def _verify_model_consistency(self, collection_name: str, requested_model_name: str):
        """
        Checks if the requested model matches the one stored in the collection's metadata.
        Raises ValueError if there is a mismatch.
        """
        collection = self.get_collection(collection_name)

        if collection.metadata and "embedding_model" in collection.metadata:
            stored_model = collection.metadata["embedding_model"]
            if stored_model != requested_model_name:
                error_msg = (
                    f"Model mismatch for collection '{collection_name}'. "
                    f"Collection was created with '{stored_model}', but you requested '{requested_model_name}'. "
                    "A collection can only be used with the model it was created with."
                )
                log.error(error_msg)
                raise ValueError(error_msg)
        else:
            log.warning(
                f"Collection '{collection_name}' has no model metadata. Skipping consistency check for backward compatibility.")

    def add_to_collection(
            self,
            collection_name: str,
            documents: list[str],
            metadatas: list[dict],
            ids: list[str],
            model: SentenceTransformer,
            model_name: str
    ):
        """
        Embeds documents using the provided model and adds them to a collection,
        after verifying model consistency.
        """
        self._verify_model_consistency(collection_name, model_name)
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
            model_name: str,
            where: Optional[dict] = None
    ) -> dict:
        """
        Embeds query texts and performs a similarity search on a collection.
        """
        self._verify_model_consistency(collection_name, model_name)
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
            model_name: str,
            documents: list[str] = None,
            metadatas: list[dict] = None
    ):
        """
        Updates documents and/or metadatas, after verifying model consistency
        if documents are being re-embedded.
        """
        if documents is not None:
            # Only check if we are re-embedding, otherwise it's not model-dependent.
            self._verify_model_consistency(collection_name, model_name)

        collection = self.get_collection(collection_name)
        log.info(f"Updating {len(ids)} documents by ID in '{collection_name}'.")

        update_kwargs = {}
        if metadatas is not None:
            update_kwargs['metadatas'] = metadatas
        if documents is not None:
            log.info(f"Re-calculating embeddings for {len(documents)} updated document(s)...")
            new_embeddings = model.encode(documents, convert_to_tensor=True).tolist()
            update_kwargs['documents'] = documents
            update_kwargs['embeddings'] = new_embeddings

        if not update_kwargs:
            log.warning("Update called with no documents or metadatas to update.")
            return

        collection.update(ids=ids, **update_kwargs)

    def delete_from_collection(self, collection_name: str, ids: list[str]):
        """Deletes documents from a collection by ID."""
        collection = self.get_collection(collection_name)
        log.info(f"Deleting {len(ids)} documents by ID from '{collection_name}'.")
        return collection.delete(ids=ids)

    def clear_collection(self, name: str):
        """Deletes all documents from a collection, but keeps the collection."""
        collection = self.get_collection(name)
        # Get all IDs in the collection. The `get()` method with no IDs returns everything.
        existing_ids = collection.get(include=[])['ids']
        if not existing_ids:
            log.info(f"Collection '{name}' is already empty. Nothing to clear.")
            return

        log.info(f"Clearing all {len(existing_ids)} documents from collection '{name}'...")
        collection.delete(ids=existing_ids)
        log.info(f"Successfully cleared collection '{name}'.")

    def export_collection(self, name: str) -> Iterator[Dict[str, Any]]:
        """
        Yields all documents from a collection for export.

        This is a generator to handle large collections without consuming too much memory.
        """
        collection = self.get_collection(name)
        log.info(f"Beginning export of collection '{name}'...")

        # The `get()` method retrieves all items if no IDs are specified.
        # We retrieve everything that can be exported.
        data = collection.get(include=["documents", "metadatas", "embeddings"])

        # The data is returned as a dict of lists, so we need to zip it
        # back into a list of individual document dicts.
        count = len(data['ids'])
        for i in range(count):
            yield {
                "id": data["ids"][i],
                "document": data["documents"][i],
                "metadata": data["metadatas"][i],
                "embedding": data["embeddings"][i],
            }
        log.info(f"Finished exporting {count} documents from '{name}'.")

    def batch_add_to_collection(
            self,
            collection_name: str,
            model_name: str,
            ids: List[str],
            documents: List[str],
            metadatas: List[Dict],
            embeddings: List[List[float]]
    ):
        """Adds a batch of items with pre-computed embeddings to a collection."""
        self._verify_model_consistency(collection_name, model_name)
        collection = self.get_collection(collection_name)
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        log.info(f"Successfully added batch of {len(documents)} documents to '{collection_name}'.")





