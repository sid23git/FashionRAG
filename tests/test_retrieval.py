"""
Basic unit tests for the Glance Fashion Retrieval pipeline.
"""

import sys
import os
# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from src.indexing.image_encoder import ImageEncoder
from src.retrieval.query_encoder import QueryEncoder


def test_image_encoder_initialization():
    encoder = ImageEncoder()
    assert encoder is not None


def test_image_encoder_output_dimensions():
    encoder = ImageEncoder()
    # Using a dummy filename
    emb = encoder.encode_image("dummy_image.jpg")
    assert emb.shape == (512,)
    # Verify it is normalized
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_query_encoder_text():
    q_encoder = QueryEncoder()
    emb = q_encoder.encode_text_query("classic leather boots")
    assert emb.shape == (512,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)
