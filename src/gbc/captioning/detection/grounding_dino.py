# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from typing import Optional
from PIL import Image
from functools import cache

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from gbc.utils import get_gbc_logger
from .detection import Detection


@cache
def load_grounding_dino(
    model_name: str = "IDEA-Research/grounding-dino-tiny", device: Optional[str] = None
):
    logger = get_gbc_logger()
    logger.info("Load GroundingDINO model...")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    return processor, model


class GroundingDinoDetection(Detection):
    """
    Detection wrapper for
    `GroudingDINO <https://github.com/IDEA-Research/GroundingDINO/>`_ model.

    .. note::
       The loaded model is cached in memory so that repeated instantiations of
       the class with the same parameters would reuse the same model.

    Attributes
    ----------
    model_name: str, default="IDEA-Research/grounding-dino-tiny"
        The name of the GroundingDINO model to use.
    device: Optional[str], default=None
        The device to use for inference.
    box_threshold: float, default=0.25
        The bbox threshold for ``processor.post_process_grounded_object_detection``.
    """

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
        box_threshold: float = 0.25,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor, self.model = load_grounding_dino(model_name, device)
        self.device = device
        self.box_threshold = box_threshold

    def detect_core(
        self, image: np.ndarray, texts: list[str]
    ) -> tuple[list[tuple[int, int, int, int]], list[float], list[int]]:

        image = Image.fromarray(image)
        texts = [text.strip(".") + "." for text in texts]
        inputs = self.processor(
            images=[image for _ in range(len(texts))],
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # The problem of GroundingDINO is that it returns probability token by token
        # We encode and compute for each text separately and take the maximum token
        # score for each piece of text
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            # This does not matter as we do not use returned label
            text_threshold=self.box_threshold,
            target_sizes=[image.size[::-1] for _ in range(len(texts))],
        )
        scores = []
        bboxes = []
        labels = []
        for label, result in enumerate(results):
            scores.extend(result["scores"].tolist())
            bboxes.extend(result["boxes"].tolist())
            labels.extend([label] * len(result["scores"]))
        return bboxes, scores, labels
