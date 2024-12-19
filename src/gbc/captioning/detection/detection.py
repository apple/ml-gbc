# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from abc import ABC
from typing import Optional, Union
from collections import defaultdict
from pydantic import BaseModel

import numpy as np
import torch
from torchvision.ops import nms

from gbc.data.bbox import Bbox, remove_contained_bboxes, select_high_conf_bboxes


class DetectionTextInput(BaseModel):
    """
    Class containing the text for detection along with its
    associated post-processing hyper-parameters.

    The default hyper-parameters are set to identify multiple bboxes
    for one piece of text
    (large :attr:`~nms_threshold` and :attr:`~remove_inside` set to False).
    Otherwise, when a hyper-parameter is set to ``None``, the corresponding feature
    is disabled.

    Attributes
    ----------
    text : str
        Input for open-vocabulary object detection.
    score_threshold : float | None, default=0.05
        Threshold on confidence score.
    nms_threshold : float | None, default=0.2
        NMS threshold.
    min_abs_area : int | None, default=5000
        Minimum acceptable absolute area of bboxes.
    max_rel_area : float | None, default=None
        Maximum acceptable relative area of bboxes.
    remove_inside : bool | None, default=False

        - If ``True``, we remove bboxes that are contained in other bboxes.
        - If ``False``, we remove bboxes that contain other bboxes.
    select_high_conf_tolerance : float | None, default=None
        When set, we only select bboxes whose confidence scores are close enough
        to the highest confidence score, as defined by the tolerance value.
    topk : int | None, default=6
        Maximum number of bboxes to keep.
    """

    text: str
    score_threshold: Optional[float] = 0.05
    min_abs_area: Optional[int] = 5000
    max_rel_area: Optional[float] = None
    nms_threshold: Optional[float] = 0.2
    remove_inside: Optional[bool] = False
    select_high_conf_tolerance: Optional[float] = None
    topk: Optional[int] = 6

    def select_bboxes(
        self,
        bboxes: list[tuple[int, int, int, int]],
        scores: list[float],
        img_width: int,
        img_height: int,
    ) -> list[Bbox]:
        """
        Selects bounding boxes based on the defined hyper-parameters.

        The function applies score thresholding, area filtering, non-maximum
        suppression (NMS), normalization of coordinates, and removes bounding
        boxes that are contained in other bounding boxes or that contain other
        bounding boxes based on the provided hyper-parameters.
        Additionally, it filters by confidence tolerance and limits the number
        of bounding boxes to keep.

        Parameters
        ----------
        bboxes
            List of bounding boxes represented as tuples (left, top, right, bottom).
        scores
            List of confidence scores for each bounding box.
        img_width
            Width of the image.
        img_height
            Height of the image.

        Returns
        -------
        list[Bbox]
            List of selected and processed bounding boxes.
        """

        # Dimension (N, 4)
        bboxes = torch.tensor(bboxes)
        # Dimension (N,)
        scores = torch.tensor(scores)

        # Apply score threshold
        keep_idxs = scores > self.score_threshold
        bboxes = bboxes[keep_idxs]
        scores = scores[keep_idxs]

        # Apply area threshold
        if self.min_abs_area is not None or self.max_rel_area is not None:
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        if self.min_abs_area is not None:
            keep_idxs = area > self.min_abs_area
            bboxes = bboxes[keep_idxs]
            scores = scores[keep_idxs]
            area = area[keep_idxs]
        if self.max_rel_area is not None:
            area = area / (img_width * img_height)
            keep_idxs = area < self.max_rel_area
            bboxes = bboxes[keep_idxs]
            scores = scores[keep_idxs]

        # Apply NMS
        if self.nms_threshold is not None:
            keep_idxs = nms(bboxes, scores, iou_threshold=self.nms_threshold)
            bboxes = bboxes[keep_idxs].tolist()
            scores = scores[keep_idxs].tolist()

        # Normalize coordinates and create Bbox instances
        converted_bboxes = []
        for bbox, score in zip(bboxes, scores):
            left, top, right, bottom = bbox
            left = left / img_width
            right = right / img_width
            top = top / img_height
            bottom = bottom / img_height
            converted_bbox = Bbox(
                left=left, top=top, right=right, bottom=bottom, confidence=score
            )
            converted_bboxes.append(converted_bbox)

        # We remove small bboxes if we only want one big bbox
        # Otherwise we remove the big bboxes that contain small bboxes
        if self.remove_inside is not None:
            converted_bboxes = remove_contained_bboxes(
                converted_bboxes, remove_inside=self.remove_inside
            )

        # Select only bboxes with confidences scores close enough
        # to the maximum confidence
        if self.select_high_conf_tolerance is not None:
            converted_bboxes = select_high_conf_bboxes(
                converted_bboxes, tolerent_diff=self.select_high_conf_tolerance
            )

        # Keep at most k bboxes with the highest confidence
        if self.topk is not None and len(converted_bboxes) > self.topk:
            converted_bboxes = sorted(
                converted_bboxes, key=lambda x: x.confidence, reverse=True
            )[: self.topk]
        return converted_bboxes


class Detection(ABC):
    """
    The abstract detection class.

    Subclasses should implement ``__init__`` for initializing the model
    and :meth:`~detect_core` for the actual detection process.
    """

    def detect_core(
        self, image: np.ndarray, texts: list[str]
    ) -> tuple[list[tuple[int, int, int, int]], list[float], list[int]]:
        """
        Perform elementary detection for a single image and a list of texts

        Parameters
        ----------
        image
            Image of size (height, width, 3).
        texts
            List of texts to detect.

        Returns
        -------
        bboxes
            List of bounding boxes, represented as (left, top, right, bottom).
        scores
            List of confidence scores.
        labels
            List of index positions of the corresponding texts as in ``texts``.
        """
        raise NotImplementedError

    def _post_process_bboxes(
        self,
        bboxes: list[tuple[int, int, int, int]],
        scores: list[float],
        labels: list[int],
        texts: list[DetectionTextInput],
        img_width: int,
        img_height: int,
    ) -> dict[str, list[Bbox]]:
        """
        Post-process the detected bounding boxes
        """
        # Group bboxes by label
        grouped_bboxes = defaultdict(list)
        for bbox, score, label in zip(bboxes, scores, labels):
            grouped_bboxes[label].append((bbox, score))

        grouped_bboxes_converted = dict()
        for label, bbox_scores in grouped_bboxes.items():
            # Unzip bbox and scores for NMS
            bboxes, scores = zip(*bbox_scores)
            try:
                text_input = texts[label]
            # Not sure why this happens
            except IndexError as e:
                print(f"IndexError with texts: {texts}")
                print(f"IndexError with label: {label}")
                raise e
            bboxes = text_input.select_bboxes(bboxes, scores, img_width, img_height)
            if len(bboxes) == 0:
                continue
            grouped_bboxes_converted[text_input.text] = bboxes
        return grouped_bboxes_converted

    def detect(
        self,
        image: np.ndarray,
        texts: list[Union[str, DetectionTextInput]],
    ) -> list[Bbox]:
        """
        Detect objects in an image based on a list of texts or
        :class:`~DetectionTextInput` objects.

        Parameters
        ----------
        image
            Image of size (height, width, 3).
        texts
            List of texts or DetectionTextInput objects to detect.

        Returns
        -------
        list[Bbox]
            List of detected Bbox objects after post-processing.
        """
        text_strings = []
        text_complete_inputs = []

        for text in texts:
            if isinstance(text, str):
                text_strings.append(text)
                text_complete_inputs.append(DetectionTextInput(text=text))
            else:
                text_strings.append(text.text)
                text_complete_inputs.append(text)
        assert len(text_strings) == len(set(text_strings)), "Texts must be unique"

        bboxes, scores, labels = self.detect_core(image, text_strings)

        # Post-process bboxes
        bboxes = self._post_process_bboxes(
            bboxes,
            scores,
            labels,
            text_complete_inputs,
            image.shape[1],
            image.shape[0],
        )
        return bboxes
