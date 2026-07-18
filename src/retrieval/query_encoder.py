"""
Module for encoding text and image queries into a joint embedding space.
"""

import os
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False


class QueryEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the Query Encoder. Loads the CLIP model if available,
        otherwise falls back to generating deterministic mock query embeddings.
        """
        self.model_name = model_name
        self.device = "cuda" if HAS_TORCH_TRANSFORMERS and torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

        if HAS_TORCH_TRANSFORMERS:
            try:
                self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.model.eval()
            except Exception as e:
                print(f"Warning: Failed to load CLIP model in query encoder ({e}). Using mock fallback.")
        else:
            print("Warning: torch/transformers not installed. QueryEncoder is in mock mode.")

    def encode_text_query(self, query_text: str) -> np.ndarray:
        """
        Encodes a natural language text query.
        """
        if self.model is not None and self.processor is not None:
            try:
                with torch.no_grad():
                    inputs = self.processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
                    text_features = self.model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    return text_features.cpu().numpy().flatten().astype("float32")
            except Exception as e:
                print(f"Error encoding text query '{query_text}': {e}. Using mock fallback.")

        # Deterministic Mock text query vector
        seed = sum(ord(c) for c in query_text)
        rng = np.random.default_rng(seed)
        mock_vector = rng.standard_normal(512).astype("float32")
        norm = np.linalg.norm(mock_vector)
        if norm > 0:
            mock_vector /= norm
        return mock_vector

    def encode_image_query(self, image_path: str) -> np.ndarray:
        """
        Encodes an image query (e.g. for image-to-image fashion search).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Query image path does not exist: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not open image query {image_path}: {e}")

        if self.model is not None and self.processor is not None:
            try:
                with torch.no_grad():
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    return image_features.cpu().numpy().flatten().astype("float32")
            except Exception as e:
                print(f"Error encoding image query {image_path}: {e}. Using mock fallback.")

        # Deterministic Mock image query vector
        seed = sum(ord(c) for c in os.path.basename(image_path))
        rng = np.random.default_rng(seed)
        mock_vector = rng.standard_normal(512).astype("float32")
        norm = np.linalg.norm(mock_vector)
        if norm > 0:
            mock_vector /= norm
        return mock_vector
