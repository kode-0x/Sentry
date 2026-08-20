"""Face detection and recognition for Sentry.

Uses OpenCV's DNN-based face detector for robust face detection
and a lightweight face embedding model (OpenCV FaceRecognizerSF)
for generating and comparing face embeddings via cosine similarity.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import cv2
import numpy as np

from src.config import get_config, resolve_path
from src.logger import get_logger

# Suppress non-critical OpenCV DNN warnings about graph engine targets
os.environ.setdefault("OPENCV_DNN_BACKEND_INFERENCE_ENGINE_TYPE", "CPU")

# OpenCV DNN face detector model files
_FACE_DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
_FACE_RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"

# Force legacy OpenCV DNN backend for ONNX compatibility with OpenCV 5.x
_DNN_BACKEND = cv2.dnn.DNN_BACKEND_OPENCV
_DNN_TARGET = cv2.dnn.DNN_TARGET_CPU


class FaceRecognitionError(Exception):
    """Raised when face recognition operations fail."""


class FaceDetector:
    """DNN-based face detector using OpenCV's YuNet model.

    Detects faces in BGR images and returns bounding boxes
    with facial landmarks.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        """Initialize the face detector.

        Args:
            model_path: Path to YuNet ONNX model. Auto-downloads if missing.

        Raises:
            FaceRecognitionError: If the model cannot be loaded.
        """
        self._logger = get_logger()
        config = get_config()
        storage = config.get("storage", {})
        model_dir = resolve_path(storage.get("model_directory", "./models"))

        if model_path is None:
            model_path = model_dir / _FACE_DETECTOR_MODEL

        if not model_path.exists():
            self._download_detector_model(model_path)

        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                (320, 320),  # Will be resized per frame
                score_threshold=0.5,
                nms_threshold=0.3,
                top_k=10,
                backend_id=_DNN_BACKEND,
                target_id=_DNN_TARGET,
            )
        except cv2.error as e:
            raise FaceRecognitionError(
                f"Failed to load face detector model: {e}"
            ) from e

    def _download_detector_model(self, dest: Path) -> None:
        """Download the YuNet face detection model.

        Args:
            dest: Destination file path.

        Raises:
            FaceRecognitionError: If download fails.
        """
        url = (
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
            "face_detection_yunet_2023mar.onnx"
        )
        self._logger.info(f"Downloading face detection model from {url}")
        try:
            import requests

            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self._logger.info("Face detection model downloaded successfully")
        except Exception as e:
            raise FaceRecognitionError(
                f"Failed to download face detection model: {e}. "
                f"Please manually download from {url} and place at {dest}"
            ) from e

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Detect faces in a BGR image.

        Args:
            frame: BGR image as NumPy array.

        Returns:
            List of face detections. Each detection is a 1D array containing:
            [x, y, w, h, ...landmarks..., confidence].
            Empty list if no faces found.
        """
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))

        _, faces = self._detector.detect(frame)

        if faces is None:
            return []

        return [faces[i] for i in range(faces.shape[0])]

    def get_face_roi(self, frame: np.ndarray, detection: np.ndarray) -> np.ndarray:
        """Extract the face region of interest from a frame.

        Args:
            frame: BGR image as NumPy array.
            detection: Single face detection array from detect().

        Returns:
            Cropped face region as BGR image.
        """
        x, y, w, h = detection[:4].astype(int)
        # Clamp to frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        return frame[y : y + h, x : x + w]


class FaceRecognizer:
    """Face embedding generator and comparator using OpenCV's SFace model.

    Generates 128-dimensional face embeddings and compares them
    using cosine similarity.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        """Initialize the face recognizer.

        Args:
            model_path: Path to SFace ONNX model. Auto-downloads if missing.

        Raises:
            FaceRecognitionError: If the model cannot be loaded.
        """
        self._logger = get_logger()
        config = get_config()
        storage = config.get("storage", {})
        model_dir = resolve_path(storage.get("model_directory", "./models"))

        if model_path is None:
            model_path = model_dir / _FACE_RECOGNIZER_MODEL

        if not model_path.exists():
            self._download_recognizer_model(model_path)

        try:
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(model_path), "",
                backend_id=_DNN_BACKEND,
                target_id=_DNN_TARGET,
            )
        except cv2.error as e:
            raise FaceRecognitionError(
                f"Failed to load face recognizer model: {e}"
            ) from e

    def _download_recognizer_model(self, dest: Path) -> None:
        """Download the SFace recognition model.

        Args:
            dest: Destination file path.

        Raises:
            FaceRecognitionError: If download fails.
        """
        url = (
            "https://huggingface.co/opencv/face_recognition_sface/resolve/main/"
            "face_recognition_sface_2021dec.onnx"
        )
        self._logger.info(f"Downloading face recognition model from {url}")
        try:
            import requests

            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self._logger.info("Face recognition model downloaded successfully")
        except Exception as e:
            raise FaceRecognitionError(
                f"Failed to download face recognition model: {e}. "
                f"Please manually download from {url} and place at {dest}"
            ) from e

    def get_embedding(
        self, frame: np.ndarray, face_detection: np.ndarray
    ) -> np.ndarray:
        """Generate a face embedding for a detected face.

        Args:
            frame: Original BGR image.
            face_detection: Face detection array from FaceDetector.detect().

        Returns:
            128-dimensional face embedding as a 1D NumPy array.

        Raises:
            FaceRecognitionError: If embedding generation fails.
        """
        try:
            aligned = self._recognizer.alignCrop(frame, face_detection)
            embedding = self._recognizer.feature(aligned)
            return embedding.flatten()
        except cv2.error as e:
            raise FaceRecognitionError(
                f"Failed to generate face embedding: {e}"
            ) from e

    def compare(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compare two face embeddings using cosine similarity.

        Args:
            embedding1: First face embedding.
            embedding2: Second face embedding.

        Returns:
            Cosine similarity score (higher = more similar).
            Typically, scores above 0.363 indicate same person
            for SFace model, but this is configurable.
        """
        # Reshape for OpenCV's match function
        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)
        score = self._recognizer.match(
            e1, e2, cv2.FaceRecognizerSF_FR_COSINE
        )
        return float(score)

    def is_match(
        self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float | None = None
    ) -> bool:
        """Check if two embeddings represent the same person.

        Args:
            embedding1: First face embedding.
            embedding2: Second face embedding.
            threshold: Similarity threshold. Defaults to config value.

        Returns:
            True if embeddings match (same person), False otherwise.
        """
        if threshold is None:
            config = get_config()
            detection_config = config.get("detection", {})
            threshold = detection_config.get("recognition_threshold", 0.55)

        score = self.compare(embedding1, embedding2)
        return score >= threshold


class EnrolledFaceStore:
    """Manages stored face embeddings for the authorized user.

    Stores multiple embeddings to account for different angles,
    lighting, and expressions. Matching is done against the best
    (highest similarity) stored embedding.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        """Initialize the face store.

        Args:
            store_path: Path to the embeddings file. Defaults to models/enrolled_face.pkl.
        """
        config = get_config()
        storage = config.get("storage", {})
        model_dir = resolve_path(storage.get("model_directory", "./models"))

        if store_path is None:
            store_path = model_dir / "enrolled_face.pkl"

        self._store_path = store_path
        self._embeddings: list[np.ndarray] = []
        self._load()

    def _load(self) -> None:
        """Load stored embeddings from disk."""
        if self._store_path.exists():
            try:
                with open(self._store_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, list) and all(
                    isinstance(e, np.ndarray) for e in data
                ):
                    self._embeddings = data
            except (pickle.UnpicklingError, EOFError, OSError):
                self._embeddings = []

    def save(self) -> None:
        """Persist embeddings to disk."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._store_path, "wb") as f:
            pickle.dump(self._embeddings, f)

    def add_embedding(self, embedding: np.ndarray) -> None:
        """Add a face embedding to the store.

        Args:
            embedding: 128-dimensional face embedding.
        """
        self._embeddings.append(embedding.copy())

    def clear(self) -> None:
        """Remove all stored embeddings."""
        self._embeddings = []
        if self._store_path.exists():
            self._store_path.unlink()

    @property
    def count(self) -> int:
        """Number of stored embeddings."""
        return len(self._embeddings)

    @property
    def is_enrolled(self) -> bool:
        """Whether at least one embedding is stored."""
        return len(self._embeddings) > 0

    def best_match_score(
        self, embedding: np.ndarray, recognizer: FaceRecognizer
    ) -> float:
        """Find the best (highest) match score against stored embeddings.

        Args:
            embedding: Face embedding to compare.
            recognizer: FaceRecognizer instance for comparison.

        Returns:
            Highest cosine similarity score across all stored embeddings.
            Returns 0.0 if no embeddings are enrolled.
        """
        if not self._embeddings:
            return 0.0

        scores = [
            recognizer.compare(embedding, stored)
            for stored in self._embeddings
        ]
        return max(scores)

    def is_authorized(
        self,
        embedding: np.ndarray,
        recognizer: FaceRecognizer,
        threshold: float | None = None,
    ) -> bool:
        """Check if a face embedding matches the enrolled user.

        Args:
            embedding: Face embedding to check.
            recognizer: FaceRecognizer instance.
            threshold: Similarity threshold. Defaults to config value.

        Returns:
            True if the face matches any enrolled embedding above threshold.
        """
        if threshold is None:
            config = get_config()
            detection_config = config.get("detection", {})
            threshold = detection_config.get("recognition_threshold", 0.55)

        best_score = self.best_match_score(embedding, recognizer)
        return best_score >= threshold
