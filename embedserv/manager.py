import asyncio
import logging
from datetime import datetime, timedelta, timezone

import torch
from sentence_transformers import SentenceTransformer

from .config import MODELS_DIR

# Set up a logger for this module
log = logging.getLogger(__name__)

# Default inactivity timeout: 5 minutes
DEFAULT_KEEP_ALIVE_SECONDS = 5 * 60


class ModelManager:
    """
    Manages the lifecycle of a single sentence-transformer model.
    """

    def __init__(self, default_keep_alive: int = DEFAULT_KEEP_ALIVE_SECONDS):
        self._current_model: SentenceTransformer | None = None
        self._current_model_name: str | None = None
        self._current_device: str | None = None  #
        self._last_used: datetime | None = None
        self._default_keep_alive_duration: timedelta = timedelta(seconds=default_keep_alive)
        self._current_keep_alive_duration: timedelta = self._default_keep_alive_duration
        self._unload_task: asyncio.Task | None = None
        self._load_lock = asyncio.Lock()

    async def load_model(self, model_name: str, device: str | None = None, keep_alive_override: int | None = None):
        """
        Loads a model into memory. If a different model is already loaded,
        it will be unloaded first. The keep-alive is set during this load.
        """
        async with self._load_lock:
            # If the same model is requested, just update its last used time.
            # If the same model on the same device is requested, do nothing.

            if self._current_model:
                self.unload_model()  # This unloads the old model/device

            log.info(f"Loading model '{model_name}' into memory...")
            try:
                # Set the keep-alive duration for this specific loading instance.
                # It will persist until the model is unloaded.
                if keep_alive_override is not None and keep_alive_override > 0:
                    self._current_keep_alive_duration = timedelta(seconds=keep_alive_override)
                    log.info(f"Setting keep-alive for '{model_name}' to {self._current_keep_alive_duration}.")
                else:
                    self._current_keep_alive_duration = self._default_keep_alive_duration

                # Load the model from disk
                self._current_model = SentenceTransformer(
                    model_name,
                    cache_folder=str(MODELS_DIR),
                    device=device
                )
                self._current_model_name = model_name
                self._current_device = str(self._current_model.device)
                log.info(f"Successfully loaded model '{model_name}' on device '{self._current_device}'.")

                # Update the timestamp
                self._update_last_used()

                # Schedule the background task to check for inactivity
                if self._unload_task:
                    self._unload_task.cancel()
                self._unload_task = asyncio.create_task(self._background_unload_check())

            except Exception as e:
                log.error(f"Failed to load model '{model_name}': {e}")
                self._current_model = None
                self._current_model_name = None
                raise e

    def get_model(self) -> SentenceTransformer | None:
        """
        Returns the currently loaded model instance and updates its last-used time.
        This does NOT change the keep-alive duration.
        """
        if self._current_model:
            self._update_last_used()
        return self._current_model

    def unload_model(self):
        """
        Unloads the model from memory, clears VRAM, and cancels the background check.
        """
        if not self._current_model:
            return

        log.info(f"Unloading model '{self._current_model_name}'.")
        model_name = self._current_model_name
        device = self._current_device
        self._current_model = None
        self._current_model_name = None
        self._current_device = None
        self._last_used = None
        # Reset keep-alive to default for the *next* model load
        self._current_keep_alive_duration = self._default_keep_alive_duration

        if self._unload_task:
            self._unload_task.cancel()
            self._unload_task = None

        # Explicitly clear GPU VRAM if CUDA is available
        if torch.cuda.is_available():
            log.info("Clearing CUDA cache to free VRAM.")
            torch.cuda.empty_cache()

        log.info(f"Model '{model_name}' on device '{device}' has been unloaded.")

    def _update_last_used(self):
        """Internal method to simply update the last used timestamp."""
        self._last_used = datetime.now(timezone.utc)

    async def _background_unload_check(self):
        """
        A background task that periodically checks if the model should be unloaded.
        """
        log.info(
            f"Starting background inactivity check for '{self._current_model_name}'. Will unload after {self._current_keep_alive_duration} of inactivity.")

        while True:
            await asyncio.sleep(30)

            if self._last_used is None:
                break

            time_since_last_use = datetime.now(timezone.utc) - self._last_used
            if time_since_last_use > self._current_keep_alive_duration:
                log.info(
                    f"Model '{self._current_model_name}' has been inactive for {time_since_last_use.total_seconds():.0f}s. Unloading.")
                self.unload_model()
                break

