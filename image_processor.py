
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


CATEGORY_PROMPTS = {
    "gate": ["an airport gate sign", "a departure gate sign", "airport departure gates"],
    "baggage_claim": ["an airport baggage claim sign", "a luggage collection sign", "an airport baggage hall sign"],
    "check_in": ["an airport check-in sign", "airline check-in desks", "airport check-in counters"],
    "security": ["an airport security sign", "an airport security checkpoint", "passenger security screening"],
    "lounge": ["an airport lounge sign", "a passenger lounge", "an airport business lounge"],
    "restaurant": ["an airport restaurant sign", "an airport food court", "airport food and dining"],
    "transport": ["an airport transport sign", "an airport train station", "airport taxi and ground transport"],
    "information": ["an airport information sign", "an airport information desk", "airport passenger information"],
    "lost_and_found": ["an airport lost and found sign", "airport lost property", "airport lost items office"],
    "special_assistance": ["an airport special assistance sign", "airport accessibility services", "passenger mobility assistance"],
}


class ImageProcessor:
    def __init__(
        self,
        kb_path,
        model_name="openai/clip-vit-base-patch32",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        with open(kb_path, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

        self.category_names = list(CATEGORY_PROMPTS)
        self.category_embeddings = self._build_category_embeddings()
        self.index = faiss.IndexFlatIP(self.category_embeddings.shape[1])
        self.index.add(self.category_embeddings)

    @staticmethod
    def _extract_features(output):
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, (tuple, list)):
            return output[0]
        raise TypeError(f"Unsupported CLIP output type: {type(output)}")

    def _build_category_embeddings(self):
        embeddings = []
        with torch.no_grad():
            for category in self.category_names:
                inputs = self.processor(
                    text=CATEGORY_PROMPTS[category],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                features = self._extract_features(self.model.get_text_features(**inputs))
                features = F.normalize(features, dim=-1)
                category_feature = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
                embeddings.append(category_feature.cpu().numpy()[0])
        return np.asarray(embeddings, dtype="float32")

    def encode_image(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self._extract_features(
                self.model.get_image_features(pixel_values=pixel_values)
            )
        return F.normalize(features, dim=-1).cpu().numpy().astype("float32")

    def classify(self, image, top_k=3):
        scores, indices = self.index.search(self.encode_image(image), top_k)
        return [
            {"category": self.category_names[index], "similarity": float(score)}
            for score, index in zip(scores[0], indices[0])
        ]

    def get_kb_candidates(self, category):
        return [record for record in self.knowledge_base if record["category"] == category]

    def analyse(self, image, top_k=3):
        predictions = self.classify(image, top_k=top_k)
        best_category = predictions[0]["category"]
        return {
            "predictions": predictions,
            "best_category": best_category,
            "kb_candidates": self.get_kb_candidates(best_category),
        }
