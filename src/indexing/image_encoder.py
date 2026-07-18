"""Image encoder module using CLIP for fashion image embeddings."""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class ImageEncoder:
    """Encodes fashion images into normalized CLIP embeddings.

    Uses the OpenAI CLIP ViT-B/32 model to produce 512-dimensional
    image embeddings suitable for similarity search and retrieval.

    Attributes:
        device: Torch device used for inference (CUDA or CPU).
        processor: CLIP image preprocessor.
        model: CLIP vision-language model.
    """

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32"
    ) -> None:
        """Initializes the ImageEncoder with a pretrained CLIP model.

        Args:
            model_name: Hugging Face model identifier for CLIP.
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", self.device)

        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name
        )
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("CLIP model '%s' loaded successfully.", model_name)

    def encode(self, image_path: str) -> np.ndarray:
        """Encodes a single image into a normalized embedding vector.

        Args:
            image_path: Filesystem path to the image file.

        Returns:
            A 1-D float32 numpy array of L2-normalized CLIP embeddings.

        Raises:
            FileNotFoundError: If the image path does not exist.
            ValueError: If the file cannot be opened as an image.
        """
        path = Path(image_path)

        if not path.is_file():
            logger.error("Image not found: %s", image_path)
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.error("Failed to open image '%s': %s", image_path, exc)
            raise ValueError(
                f"Cannot open '{image_path}' as an image."
            ) from exc

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            embedding = self.model.visual_projection(
                vision_outputs.pooler_output
            )

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().astype(np.float32)[0]