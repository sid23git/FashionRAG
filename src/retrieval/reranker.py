"""Attribute-aware reranking module for fashion image retrieval."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class AttributeReranker:
    """Re-ranks FAISS search results using metadata attributes.
    
    Computes a simple attribute match score based on query keywords
    and the image path from metadata, then combines it with the CLIP score.
    """

    def __init__(self, metadata_path: str = "indexes/image_metadata.pkl") -> None:
        """Initializes the AttributeReranker.

        Args:
            metadata_path: Path to the pickled image metadata.
        """
        if not Path(metadata_path).is_file():
            logger.error("Metadata file not found: %s", metadata_path)
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, "rb") as f:
            self.metadata: List[Tuple[int, str]] = pickle.load(f)
            
        # Create a quick lookup for metadata by image_id
        self.metadata_dict: Dict[int, str] = {
            img_id: img_path for img_id, img_path in self.metadata
        }
        logger.info("AttributeReranker initialized with %d metadata records.", len(self.metadata))

    def _compute_attribute_score(self, query: str, image_path: str) -> float:
        """Computes a simple attribute score based on query keywords.
        
        Args:
            query: The text query.
            image_path: The image path from metadata.
            
        Returns:
            A float score between 0.0 and 1.0.
        """
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
            
        path_lower = image_path.lower()
        matches = sum(1 for word in query_words if word in path_lower)
        
        return matches / len(query_words)

    def rerank(self, query: str, results: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Re-ranks the initial search results.
        
        Combines the original CLIP score with an attribute score:
        final_score = 0.8 * clip_score + 0.2 * attribute_score
        
        Args:
            query: The original search query.
            results: The list of top-K results from TextRetriever.
            
        Returns:
            A new list of re-ranked results.
        """
        if not results:
            return []

        reranked_results = []
        for res in results:
            image_id = int(res["image_id"])
            clip_score = float(res["score"])
            image_path = self.metadata_dict.get(image_id, "")
            
            attribute_score = self._compute_attribute_score(query, image_path)
            final_score = 0.8 * clip_score + 0.2 * attribute_score
            
            reranked_res = dict(res)
            reranked_res["original_score"] = clip_score
            reranked_res["attribute_score"] = attribute_score
            reranked_res["score"] = final_score
            reranked_results.append(reranked_res)
            
        # Sort by final_score descending
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Update ranks
        for rank, res in enumerate(reranked_results, start=1):
            res["rank"] = rank
            
        logger.info("Re-ranked %d results for query '%s'.", len(reranked_results), query)
        return reranked_results
