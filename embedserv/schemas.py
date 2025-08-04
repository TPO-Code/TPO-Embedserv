from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any

# helper class for DRY (Don't Repeat Yourself) principle
class ModelTaskRequest(BaseModel):
    model: str = Field(..., description="The name of the model to use for this task.")
    device: Optional[str] = Field(None, description="The device to run the model on (e.g., 'cpu', 'cuda', 'cuda:1'). Defaults to auto-detection.")

class EmbeddingRequestOptions(BaseModel):
    """Options for an embedding request."""
    keep_alive: Optional[int] = Field(None, description="Time in seconds to keep the model loaded after this request.")

class EmbeddingRequest(BaseModel):
    """Request body for the /embeddings endpoint."""
    model: str = Field(..., description="The name of the model to use for embeddings (e.g., 'all-MiniLM-L6-v2').")
    input: Union[str, List[str]] = Field(..., description="The input text or list of texts to embed.")
    device: Optional[str] = Field(None, description="The device to run on.")
    options: Optional[EmbeddingRequestOptions] = None

class Embedding(BaseModel):
    """Represents a single embedding vector."""
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0  # Note: Sentence-transformers don't give us prompt tokens easily.
    total_tokens: int = 0   # We will leave these as placeholders for now.

class EmbeddingResponse(BaseModel):
    """Response body for the /embeddings endpoint."""
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: EmbeddingUsage

class ModelList(BaseModel):
    """Response body for listing local models."""
    data: List[str]

class PullRequest(BaseModel):
    """Request body for pulling a model."""
    model: str = Field(..., description="The name of the model to pull from Hugging Face.")

class StatusResponse(BaseModel):
    """Generic status response."""
    status: str
    message: Optional[str] = None


# --- Vector DB Schemas ---

class CollectionRequest(BaseModel):
    name: str = Field(..., description="The name for the new collection.")

class CollectionCountResponse(BaseModel):
    name: str
    count: int

class GetByIdRequest(BaseModel):
    ids: List[str] = Field(..., description="A list of document IDs to retrieve.")
    # This aligns with Chroma's API for what data to return
    include: List[str] = Field(default=["metadatas", "documents"], description="Data to include in the response.")

class UpdateRequest(ModelTaskRequest):
    ids: List[str] = Field(...)
    # Documents and metadatas are optional; users might only want to update one or the other.
    documents: Optional[List[str]] = Field(None)
    metadatas: Optional[List[Dict[str, Any]]] = Field(None)

class DeleteByIdRequest(BaseModel):
    ids: List[str] = Field(..., description="A list of document IDs to delete.")

class AddRequest(ModelTaskRequest):
    documents: List[str] = Field(..., description="A list of documents to add.")
    metadatas: List[Dict[str, Any]] = Field(..., description="A list of metadata dictionaries, one for each document.")
    ids: List[str] = Field(..., description="A list of unique string IDs, one for each document.")


class QueryRequest(ModelTaskRequest):
    query_texts: List[str] = Field(..., description="A list of texts to search for.")
    n_results: int = Field(10, description="The number of results to return.")
    where: Optional[Dict[str, Any]] = Field(None, description="A metadata filter dictionary.")

class QueryResult(BaseModel):
    ids: List[List[str]]
    distances: List[List[float]]
    metadatas: List[List[Optional[Dict[str, Any]]]]
    embeddings: Optional[List[List[List[float]]]] = None
    documents: List[List[Optional[str]]]

class QueryResponse(BaseModel):
    results: QueryResult

class CollectionListResponse(BaseModel):
    collections: List[str]

class GetResponse(BaseModel):
    """Response for getting documents by ID."""
    ids: List[str]
    embeddings: Optional[List[List[float]]] = None
    metadatas: Optional[List[Dict[str, Any]]] = None
    documents: Optional[List[str]] = None


class SimilarityRequest(BaseModel):
    """Request body for the /similarity endpoint."""
    embeddings_a: List[List[float]] = Field(..., description="The first list of embedding vectors.")
    embeddings_b: List[List[float]] = Field(..., description="The second list of embedding vectors.")

class SimilarityResponse(BaseModel):
    """Response body for the /similarity endpoint."""
    similarity_scores: List[List[float]]