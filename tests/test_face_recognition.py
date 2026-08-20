"""Tests for src/face_recognition.py."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.face_recognition import EnrolledFaceStore, FaceRecognitionError


class TestEnrolledFaceStore:
    def test_empty_store_not_enrolled(self, isolated_config):
        store = EnrolledFaceStore()
        assert not store.is_enrolled
        assert store.count == 0

    def test_add_embedding(self, isolated_config, fake_embedding):
        store = EnrolledFaceStore()
        store.add_embedding(fake_embedding)
        assert store.count == 1
        assert store.is_enrolled

    def test_save_and_reload(self, isolated_config, fake_embedding):
        store = EnrolledFaceStore()
        store.add_embedding(fake_embedding)
        store.save()

        # Create a new store instance (should load from disk)
        store2 = EnrolledFaceStore()
        assert store2.count == 1
        assert store2.is_enrolled

    def test_clear(self, isolated_config, fake_embedding):
        store = EnrolledFaceStore()
        store.add_embedding(fake_embedding)
        store.save()
        store.clear()
        assert store.count == 0
        assert not store.is_enrolled

    def test_best_match_score_no_embeddings(self, isolated_config, fake_embedding):
        store = EnrolledFaceStore()
        mock_recognizer = MagicMock()
        score = store.best_match_score(fake_embedding, mock_recognizer)
        assert score == 0.0

    def test_best_match_score_with_embeddings(self, isolated_config, fake_embedding):
        store = EnrolledFaceStore()
        store.add_embedding(fake_embedding)

        mock_recognizer = MagicMock()
        mock_recognizer.compare.return_value = 0.85

        score = store.best_match_score(fake_embedding, mock_recognizer)
        assert score == 0.85
        mock_recognizer.compare.assert_called_once()

    def test_is_authorized_above_threshold(self, isolated_config, fake_embedding):
        store = EnrolledFaceStore()
        store.add_embedding(fake_embedding)

        mock_recognizer = MagicMock()
        mock_recognizer.compare.return_value = 0.8

        assert store.is_authorized(fake_embedding, mock_recognizer, threshold=0.55)

    def test_is_authorized_below_threshold(
        self, isolated_config, fake_embedding, different_embedding
    ):
        store = EnrolledFaceStore()
        store.add_embedding(fake_embedding)

        mock_recognizer = MagicMock()
        mock_recognizer.compare.return_value = 0.2

        assert not store.is_authorized(
            different_embedding, mock_recognizer, threshold=0.55
        )

    def test_best_match_multiple_embeddings(self, isolated_config):
        store = EnrolledFaceStore()
        rng = np.random.default_rng(10)

        for _ in range(5):
            emb = rng.standard_normal(128).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            store.add_embedding(emb)

        assert store.count == 5

        mock_recognizer = MagicMock()
        mock_recognizer.compare.side_effect = [0.3, 0.4, 0.9, 0.2, 0.1]

        test_emb = rng.standard_normal(128).astype(np.float32)
        score = store.best_match_score(test_emb, mock_recognizer)
        assert score == 0.9

    def test_handles_corrupted_store(self, isolated_config):
        """Store should handle corrupted pickle gracefully."""
        config_path = Path(isolated_config) / "models" / "enrolled_face.pkl"
        config_path.write_bytes(b"corrupted data")

        store = EnrolledFaceStore()
        assert store.count == 0
        assert not store.is_enrolled
