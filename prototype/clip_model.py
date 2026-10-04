import numpy as np
import torch

from PIL import Image

from transformers import (
    CLIPModel,
    CLIPProcessor
)

from config import (
    DEVICE,
    MODEL_NAME
)


class CLIPEncoder:

    def __init__(self):

        print(
            f"Loading CLIP model on {DEVICE}..."
        )

        self.processor = (
            CLIPProcessor.from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            CLIPModel.from_pretrained(
                MODEL_NAME
            ).to(DEVICE)
        )

        self.model.eval()

    # ========================================================
    # Image embeddings
    # ========================================================

    def encode_images(
        self,
        images: list[Image.Image]
    ) -> np.ndarray:
        """
        Encode multiple images at once.

        Returns:
            numpy array of shape:

            (number_of_images, 512)
        """

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            # Directly use CLIP's vision model.
            vision_outputs = (
                self.model.vision_model(
                    pixel_values=inputs[
                        "pixel_values"
                    ]
                )
            )

            pooled_output = (
                vision_outputs.pooler_output
            )

            features = (
                self.model.visual_projection(
                    pooled_output
                )
            )

        # L2 normalize
        features = (
            features /
            features.norm(
                dim=-1,
                keepdim=True
            )
        )

        return (
            features
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # ========================================================
    # Single image
    # ========================================================

    def encode_image(
        self,
        image: Image.Image
    ) -> np.ndarray:

        return self.encode_images(
            [image]
        )[0]

    # ========================================================
    # Text embedding
    # ========================================================

    def encode_text(
        self,
        text: str
    ) -> np.ndarray:
        """
        Convert text into a normalized CLIP embedding.
        """

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            text_outputs = (
                self.model.text_model(
                    input_ids=inputs[
                        "input_ids"
                    ],
                    attention_mask=inputs[
                        "attention_mask"
                    ]
                )
            )

            pooled_output = (
                text_outputs.pooler_output
            )

            features = (
                self.model.text_projection(
                    pooled_output
                )
            )

        # L2 normalize
        features = (
            features /
            features.norm(
                dim=-1,
                keepdim=True
            )
        )

        return (
            features
            .cpu()
            .numpy()
            .astype(np.float32)
        )