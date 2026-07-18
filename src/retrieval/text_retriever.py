"""Text-based retrieval module for fashion image search.

Encodes natural language queries with CLIP and searches a prebuilt
FAISS IndexFlatIP index to return the most similar fashion images.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

logger = logging.getLogger(__name__)


class TextRetriever:
    """Retrieves fashion images from a FAISS index using text queries.

    Loads a pretrained CLIP model to encode text queries into the same
    embedding space as the indexed images, then performs inner-product
    search to find the most similar results.

    Attributes:
        device: Torch device used for inference (CUDA or CPU).
        processor: CLIP text/image preprocessor.
        model: CLIP vision-language model.
        index: FAISS IndexFlatIP loaded from disk.
        metadata: List of (id, image_path) tuples.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        index_path: str = "indexes/fashion.index",
        metadata_path: str = "indexes/image_metadata.pkl",
    ) -> None:
        """Initializes the TextRetriever.

        Args:
            model_name: Hugging Face model identifier for CLIP.
            index_path: Path to the saved FAISS index file.
            metadata_path: Path to the pickled image metadata.
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", self.device)

        # --- Load CLIP model ---
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name
        )
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("CLIP model '%s' loaded successfully.", model_name)

        # --- Load FAISS index ---
        if not Path(index_path).is_file():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}. "
                "Run 'python main.py --build-index' first."
            )
        self.index: faiss.IndexFlatIP = faiss.read_index(index_path)
        logger.info(
            "FAISS index loaded from '%s' (%d vectors).",
            index_path,
            self.index.ntotal,
        )

        # --- Load metadata ---
        if not Path(metadata_path).is_file():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                "Run 'python main.py --build-index' first."
            )
        with open(metadata_path, "rb") as f:
            self.metadata: List[tuple[int, str]] = pickle.load(f)
        logger.info("Loaded metadata for %d images.", len(self.metadata))

    def _encode_text(self, query: str) -> np.ndarray:
        """Encodes a text query into a normalized CLIP embedding.

        Args:
            query: Natural language search query.

        Returns:
            A 2-D float32 numpy array of shape (1, dim) with
            L2-normalized CLIP text embeddings.
        """
        inputs = self.processor(
            text=[query], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            embedding = self.model.text_projection(
                text_outputs.pooler_output
            )

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype(np.float32)

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, object]]:
        """Searches the FAISS index for images matching a text query.

        Args:
            query: Natural language search query.
            top_k: Number of top results to return.

        Returns:
            A list of dicts, each containing:
                - rank (int): 1-based rank.
                - score (float): Cosine similarity score.
                - image_path (str): Filesystem path to the matched image.
                - image_id (int): Integer ID of the matched image.
        """
        query_embedding = self._encode_text(query)
        faiss.normalize_L2(query_embedding)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Dict[str, object]] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:
                continue
            image_id, image_path = self.metadata[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "image_path": image_path,
                "image_id": image_id,
            })

        logger.info(
            "Query '%s' returned %d results.", query, len(results)
        )
        return results
