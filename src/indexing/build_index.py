"""Offline indexing pipeline for fashion image retrieval.

Scans a directory for images, encodes them with CLIP, builds a FAISS
IndexFlatIP index, and persists the index along with image metadata.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from src.indexing.image_encoder import ImageEncoder

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}


def _discover_images(data_dir: str) -> List[Path]:
    """Recursively discovers supported image files under a directory.

    Args:
        data_dir: Root directory to scan for images.

    Returns:
        Sorted list of paths to discovered image files.
    """
    root = Path(data_dir)
    if not root.is_dir():
        logger.error("Data directory does not exist: %s", data_dir)
        return []

    images: List[Path] = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    images.sort()
    logger.info("Discovered %d images in '%s'.", len(images), data_dir)
    return images


def build_index(
    data_dir: str = "data/raw_images",
    index_path: str = "indexes/fashion.index",
    metadata_path: str = "indexes/image_metadata.pkl",
) -> None:
    """Builds a FAISS index from all images in a directory.

    Encodes every valid image with the ImageEncoder, constructs a FAISS
    IndexFlatIP index (inner-product on L2-normalised vectors equals
    cosine similarity), and saves the index and metadata to disk.

    Args:
        data_dir: Root directory containing raw images.
        index_path: Destination path for the FAISS index file.
        metadata_path: Destination path for the image metadata pickle.
    """
    # --- 1. Discover images ---
    image_paths = _discover_images(data_dir)
    if not image_paths:
        logger.warning("No images found. Aborting index build.")
        return

    # --- 2. Encode images ---
    encoder = ImageEncoder()
    embeddings: List[np.ndarray] = []
    metadata: List[Tuple[int, str]] = []
    skipped: int = 0

    for path in tqdm(image_paths, desc="Encoding images", unit="img"):
        try:
            embedding = encoder.encode(str(path))
            idx = len(metadata)
            embeddings.append(embedding)
            metadata.append((idx, str(path)))
        except (FileNotFoundError, ValueError):
            skipped += 1
            logger.warning("Skipped corrupted/invalid image: %s", path)

    if not embeddings:
        logger.error("No valid embeddings produced. Aborting index build.")
        return

    logger.info(
        "Encoded %d images successfully (%d skipped).",
        len(embeddings),
        skipped,
    )

    # --- 3. Stack and normalise embeddings ---
    matrix = np.vstack(embeddings).astype(np.float32)
    faiss.normalize_L2(matrix)
    dimension: int = matrix.shape[1]
    logger.info("Embedding matrix shape: %s", matrix.shape)

    # --- 4. Build FAISS IndexFlatIP ---
    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)
    logger.info("FAISS index built with %d vectors.", index.ntotal)

    # --- 5. Save index ---
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    logger.info("FAISS index saved to '%s'.", index_path)

    # --- 6. Save metadata ---
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info("Image metadata saved to '%s'.", metadata_path)

    logger.info("Index build complete.")
