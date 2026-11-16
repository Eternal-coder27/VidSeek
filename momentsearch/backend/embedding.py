import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from config import MODEL_ID


def _to_tensor(outputs: object) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        return outputs.text_embeds
    return outputs[0]  # type: ignore[index]


class EmbeddingService:
    def __init__(self, model_id: str = MODEL_ID):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model_id = model_id
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None

    def load(self) -> None:
        if self.model is None or self.processor is None:
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        self.load()
        assert self.model is not None and self.processor is not None
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        features = _to_tensor(outputs)
        vec = np.float32(features.detach().cpu().numpy())
        faiss.normalize_L2(vec)
        return vec

    def encode_text(self, text: str) -> np.ndarray:
        self.load()
        assert self.model is not None and self.processor is not None
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        features = _to_tensor(outputs)
        vec = np.float32(features.detach().cpu().numpy())
        faiss.normalize_L2(vec)
        return vec


embedding_service = EmbeddingService()