import logging
import asyncio
import itertools
from functools import partial
from dataclasses import dataclass, field
from typing import List, Callable, Any, Dict
from datetime import timezone

import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from .schemas import (
    EmbeddingRequest, EmbeddingResponse, Embedding, EmbeddingUsage,
    ModelList, PullRequest, StatusResponse, CollectionRequest, AddRequest,
    QueryRequest, QueryResponse, CollectionListResponse, UpdateRequest,
    GetByIdRequest, DeleteByIdRequest, CollectionCountResponse, GetResponse, SimilarityRequest, SimilarityResponse,
    ServerStatusResponse
)
from .manager import ModelManager, DEFAULT_KEEP_ALIVE_SECONDS
from .models import pull_model as pull_model_sync, list_local_models, delete_model as delete_model_sync
from .db import VectorDBManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
background_tasks = set()
# --- Queue and Worker Implementation ---

# A counter to ensure FIFO for jobs with the same priority (tie-breaker)
job_counter = itertools.count()


@dataclass(order=True)
class QueuedJob:
    """A dataclass to hold all information about a job to be processed."""
    future: asyncio.Future = field(compare=False)
    work_function: Callable = field(compare=False)
    model_name: str = field(compare=False)
    device: str | None = field(compare=False)
    args: tuple = field(compare=False, default_factory=tuple)
    kwargs: Dict[str, Any] = field(compare=False, default_factory=dict)
    keep_alive_override: int | None = field(compare=False, default=None)


# --- Global State ---
manager: ModelManager
db_manager: VectorDBManager
request_queue: asyncio.PriorityQueue
worker_task: asyncio.Task

app = FastAPI(
    title="EmbedServ",
    description="A stateless, on-demand sentence transformer server with request queueing.",
    version="0.3.0",
)


# --- MODIFIED: process_request_queue is now more robust ---
async def process_request_queue():
    """The main worker loop that processes jobs from the priority queue."""
    log.info("Request queue worker started.")
    while True:
        try:
            # This outer try block handles getting a job from the queue.
            # If this `await` is cancelled, the `CancelledError` is caught below and the loop breaks.
            priority, _, job = await request_queue.get()

            # This inner try/except/finally block handles processing the job.
            # It ensures that task_done() is called for every job that is successfully dequeued.
            try:
                log.info(f"Dequeued job (priority: {priority}, model: '{job.model_name}')")

                # 1. Load the model if it's not the correct one.
                if (manager._current_model_name != job.model_name or
                        manager._current_device != job.device):
                    await manager.load_model(
                        job.model_name,
                        device=job.device,
                        keep_alive_override=job.keep_alive_override
                    )

                model = manager.get_model()
                if model is None:
                    raise RuntimeError(f"Model '{job.model_name}' failed to load after queueing.")

                # 2. Execute the work in a thread to avoid blocking the event loop.
                job.kwargs['model'] = model
                work_to_run = partial(job.work_function, *job.args, **job.kwargs)
                result = await asyncio.to_thread(work_to_run)

                # 3. Set the result on the future to unblock the waiting endpoint.
                if not job.future.done():
                    job.future.set_result(result)

            except Exception as e:
                log.error(f"Error processing queued job for model '{job.model_name}': {e}", exc_info=True)
                if not job.future.done():
                    job.future.set_exception(e)
            finally:
                # This is only reached if get() was successful, and it guarantees
                # that we mark the task as done for the queue.
                request_queue.task_done()

        except asyncio.CancelledError:
            log.info("Request queue worker is shutting down.")
            break  # Exit the loop gracefully.

        except Exception as e:
            # This would catch a non-cancellation error from get() itself.
            log.error(f"FATAL: Unhandled error in request queue worker loop: {e}", exc_info=True)
            await asyncio.sleep(1) # Prevent a tight loop of failures.


@app.on_event("startup")
def on_startup():
    global manager, db_manager, request_queue, worker_task
    keep_alive = getattr(app.state, "keep_alive_seconds", DEFAULT_KEEP_ALIVE_SECONDS)
    manager = ModelManager(default_keep_alive=keep_alive)
    db_manager = VectorDBManager()
    request_queue = asyncio.PriorityQueue()
    worker_task = asyncio.create_task(process_request_queue())


@app.on_event("shutdown")
def on_shutdown():
    if 'worker_task' in globals() and not worker_task.done():
        worker_task.cancel()
    if 'manager' in globals():
        manager.unload_model()


# --- Helper function to queue jobs ---
async def _queue_job_and_wait(
        work_function: Callable,
        model_name: str,
        device: str | None,
        keep_alive: int | None = None,
        args: tuple = (),
        kwargs: dict = None
):
    """Helper to create a job, put it in the queue, and await the result."""
    if kwargs is None:
        kwargs = {}

    # Priority 0 (high) for a match, 1 (low) for a mismatch
    priority = 0 if (manager._current_model_name == model_name and manager._current_device == device) else 1

    future = asyncio.get_running_loop().create_future()

    job = QueuedJob(
        future=future,
        work_function=work_function,
        model_name=model_name,
        device=device,
        keep_alive_override=keep_alive,
        args=args,
        kwargs=kwargs
    )

    await request_queue.put((priority, next(job_counter), job))

    # Await the result from the worker. This will re-raise any exceptions.
    result = await future
    return result


# --- Synchronous work functions to run in a thread ---

def _work_create_embeddings(
        inputs: List[str],
        model: SentenceTransformer,
        request_model_name: str
) -> EmbeddingResponse:
    """The actual work for the embeddings endpoint."""
    vectors = model.encode(inputs, convert_to_numpy=True)
    embedding_data = [Embedding(embedding=vector.tolist(), index=i) for i, vector in enumerate(vectors)]
    return EmbeddingResponse(data=embedding_data, model=request_model_name, usage=EmbeddingUsage())

def _work_calculate_similarity(embeddings_a: List[List[float]], embeddings_b: List[List[float]]) -> List[List[float]]:
    """Calculates cosine similarity between two sets of embeddings."""
    try:
        tensor_a = torch.tensor(embeddings_a)
        tensor_b = torch.tensor(embeddings_b)
        similarity_matrix = cos_sim(tensor_a, tensor_b)
        return similarity_matrix.tolist()
    except Exception as e:
        log.error(f"Error during similarity calculation: {e}")
        # Re-raise a ValueError to be handled by the endpoint, providing a clear message.
        raise ValueError(f"Could not calculate similarity. Check if embedding dimensions match. Original error: {e}")

# --- API Endpoints ---

@app.get("/")
async def read_root(): # Changed to async for consistency
    return {"message": "EmbedServ is running. See /docs for API details."}

@app.get("/health", status_code=200)
async def health_check():
    """A simple health check endpoint for automated systems."""
    return Response(status_code=200)


@app.get("/api/v1/status", response_model=ServerStatusResponse)
async def get_server_status():
    """Provides a detailed status of the server's state."""
    # Ensure last_used is timezone-aware if it exists
    last_used = manager._last_used
    if last_used and last_used.tzinfo is None:
        last_used = last_used.replace(tzinfo=timezone.utc)

    return ServerStatusResponse(
        status="running",
        current_model=manager._current_model_name,
        current_device=manager._current_device,
        last_used_at=last_used,
        keep_alive_seconds=manager._current_keep_alive_duration.total_seconds(),
        pending_queue_jobs=request_queue.qsize()
    )
# --- Model-Dependent Endpoints ---

@app.post("/api/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    try:
        input_texts = [request.input] if isinstance(request.input, str) else request.input
        result = await _queue_job_and_wait(
            work_function=_work_create_embeddings,
            model_name=request.model,
            device=request.device,
            keep_alive=request.options.keep_alive if request.options else None,
            kwargs={'inputs': input_texts, 'request_model_name': request.model}
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/db/{collection_name}/add", response_model=StatusResponse)
async def add_to_db_collection(collection_name: str, request: AddRequest):
    try:
        await _queue_job_and_wait(
            work_function=db_manager.add_to_collection,
            model_name=request.model,
            device=request.device,
            args=(collection_name,),
            kwargs={'documents': request.documents, 'metadatas': request.metadatas, 'ids': request.ids}
        )
        return StatusResponse(status="success",
                              message=f"Added {len(request.documents)} documents to '{collection_name}'.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/db/{collection_name}/query", response_model=QueryResponse)
async def query_db_collection(collection_name: str, request: QueryRequest):
    try:
        results = await _queue_job_and_wait(
            work_function=db_manager.query_collection,
            model_name=request.model,
            device=request.device,
            args=(collection_name,),
            kwargs={'query_texts': request.query_texts, 'n_results': request.n_results, 'where': request.where}
        )
        return QueryResponse(results=results)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/db/{collection_name}/update", response_model=StatusResponse)
async def update_in_db_collection(collection_name: str, request: UpdateRequest):
    try:
        await _queue_job_and_wait(
            work_function=db_manager.update_in_collection,
            model_name=request.model,
            device=request.device,
            args=(collection_name,),
            kwargs={'ids': request.ids, 'documents': request.documents, 'metadatas': request.metadatas}
        )
        return StatusResponse(status="success", message=f"Updated {len(request.ids)} documents in '{collection_name}'.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Model-Independent Endpoints) ---

@app.post("/api/v1/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """Calculates cosine similarity between two sets of embeddings."""
    try:
        # Run the potentially heavy computation in a separate thread
        scores = await asyncio.to_thread(
            _work_calculate_similarity,
            request.embeddings_a,
            request.embeddings_b
        )
        return SimilarityResponse(similarity_scores=scores)
    except ValueError as e:
        # Catch errors from the worker (e.g., dimension mismatch)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/api/v1/db/{collection_name}/count", response_model=CollectionCountResponse)
async def get_db_collection_count(collection_name: str):
    try:
        # Run blocking I/O in a thread to not block the event loop
        count = await asyncio.to_thread(db_manager.count_documents_in_collection, collection_name)
        return CollectionCountResponse(name=collection_name, count=count)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/db/{collection_name}/get", response_model=GetResponse)
async def get_from_db_collection(collection_name: str, request: GetByIdRequest):
    try:
        # Run blocking I/O in a thread
        results = await asyncio.to_thread(db_manager.get_from_collection, collection_name, ids=request.ids, include=request.include)
        return GetResponse(**results) # Explicitly construct for clarity
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/db/{collection_name}/delete", response_model=StatusResponse)
async def delete_from_db_collection(collection_name: str, request: DeleteByIdRequest):
    try:
        # Run blocking I/O in a thread
        await asyncio.to_thread(db_manager.delete_from_collection, collection_name, ids=request.ids)
        return StatusResponse(status="success",
                              message=f"Delete request for {len(request.ids)} documents sent to '{collection_name}'.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/unload", response_model=StatusResponse)
async def unload_current_model():
    try:
        # Unloading can involve I/O or CUDA syncs, so it's safer in a thread
        await asyncio.to_thread(manager.unload_model)
        return StatusResponse(status="success", message="Model unloaded successfully.")
    except Exception as e:
        log.error(f"Error during model unload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/db", response_model=CollectionListResponse)
async def list_db_collections():
    # Run blocking I/O in a thread
    collections = await asyncio.to_thread(db_manager.list_collections)
    return CollectionListResponse(collections=collections)


@app.post("/api/v1/db", response_model=StatusResponse)
async def create_db_collection(request: CollectionRequest):
    try:
        # Run blocking I/O in a thread
        await asyncio.to_thread(db_manager.create_collection, request.name)
        return StatusResponse(status="success", message=f"Collection '{request.name}' created.")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/db/{collection_name}", response_model=StatusResponse)
async def delete_db_collection(collection_name: str):
    try:
        # Run blocking I/O in a thread
        await asyncio.to_thread(db_manager.delete_collection, collection_name)
        return StatusResponse(status="success", message=f"Collection '{collection_name}' deleted.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models", response_model=ModelList)
async def list_models():
    # Run blocking filesystem I/O in a thread
    local_models = await asyncio.to_thread(list_local_models)
    return ModelList(data=local_models)


@app.post("/api/v1/pull", response_model=StatusResponse)
async def pull_model(request: PullRequest):
    try:
        # Create a background task to run the download
        task = asyncio.create_task(asyncio.to_thread(pull_model_sync, request.model))

        # Add the task to the set. This is a good practice to prevent
        # the task from being garbage-collected prematurely.
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

        # Return immediately
        return StatusResponse(status="accepted", message=f"Pull request for model '{request.model}' accepted and is running in the background.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/models/{model_name:path}", response_model=StatusResponse)
async def delete_model(model_name: str):
    # Run blocking filesystem I/O in a thread
    deleted = await asyncio.to_thread(delete_model_sync, model_name)
    if deleted:
        return StatusResponse(status="success", message=f"Model '{model_name}' deleted.")
    else:
        # The underlying function prints errors, so we just need to return the HTTP error
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or could not be deleted.")