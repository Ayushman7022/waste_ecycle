"""Zero-shot battery vs wire scoring using a pretrained CLIP model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open_clip
import torch
from PIL import Image

CLASS_PROMPTS = {
    "battery": (
        "a close-up photo of a battery",
        "a photo of an AA battery or lithium battery",
        "a photo of a 9V battery or button cell",
    ),
    "wire": (
        "a close-up photo of an electrical wire",
        "a photo of a cable or copper wire",
        "a photo of a power cord or USB cable",
    ),
}

# A third object class, so a cup or a phone can win instead of being forced
# into battery or wire.
OTHER_PROMPTS = (
    "a close-up photo of some other object",
    "a photo of a cup, bottle, phone, book, box, or pen",
    "a photo of a hand, tool, or household item that is not a battery or a wire",
)

EMPTY_PROMPTS = (
    "a photo of an empty table or desk with nothing on it",
    "a photo of a blank wall, floor, or plain empty surface",
    "a blurry photo of an empty room",
)

DEFAULT_THRESHOLD = 0.60

# The openai weights were trained with QuickGELU; the plain ViT-B-32 config is not.
MODEL_NAME = "ViT-B-32-quickgelu"


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float
    scores: dict[str, float]


class WasteClassifier:
    """Scores an image crop as battery or wire. Hands are not treated as background."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained="openai",
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.class_names = tuple(CLASS_PROMPTS)
        self._text_features = self._encode_prompts(
            [CLASS_PROMPTS[name] for name in self.class_names] + [OTHER_PROMPTS, EMPTY_PROMPTS]
        )

    def _encode_prompts(self, prompt_groups: list[tuple[str, ...]]) -> torch.Tensor:
        class_features: list[torch.Tensor] = []
        with torch.no_grad():
            for prompts in prompt_groups:
                tokens = self.tokenizer(list(prompts)).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                class_feat = features.mean(dim=0, keepdim=True)
                class_feat = class_feat / class_feat.norm(dim=-1, keepdim=True)
                class_features.append(class_feat)
        return torch.cat(class_features, dim=0)

    def classify(
        self,
        frame_bgr: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Classification:
        if frame_bgr.size == 0 or min(frame_bgr.shape[:2]) < 8:
            return Classification(label="background", confidence=0.0, scores={})

        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ self._text_features.T)[0]
            probs = (100.0 * sims).softmax(dim=-1).detach().cpu().numpy()

        return interpret_scores(probs, self.class_names, threshold)


def interpret_scores(
    probs: np.ndarray,
    class_names: tuple[str, ...],
    threshold: float,
) -> Classification:
    """Map softmax bins (battery, wire, other, empty) to a label.

    Empty wins only when it beats every object bin. Anything that beats both
    battery and wire is invalid, even below the battery/wire threshold.
    """
    other_index = len(class_names)
    empty_index = other_index + 1
    object_probs = probs[:empty_index]
    empty_prob = float(probs[empty_index])
    scores = {name: float(probs[i]) for i, name in enumerate(class_names)}
    scores["invalid"] = float(probs[other_index])

    object_total = float(object_probs.sum())
    best_idx = int(np.argmax(object_probs))
    confidence = float(object_probs[best_idx] / object_total) if object_total > 0 else 0.0

    if empty_prob > float(np.max(object_probs)):
        return Classification(label="background", confidence=confidence, scores=scores)
    if best_idx == other_index:
        return Classification(label="invalid", confidence=confidence, scores=scores)
    if confidence < threshold:
        return Classification(label="background", confidence=confidence, scores=scores)
    return Classification(
        label=class_names[best_idx],
        confidence=confidence,
        scores=scores,
    )
