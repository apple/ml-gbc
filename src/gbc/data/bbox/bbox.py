# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import numpy as np
from typing import Optional
from pydantic import BaseModel


class Bbox(BaseModel):
    """
    The coordinates represent the relative position within the image
    """

    left: float
    top: float
    right: float
    bottom: float
    confidence: Optional[float] = None

    def compute_area(self) -> float:
        # Calculate the area of a bounding box
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    def compute_overlap_area(self, another_bbox: "Bbox") -> float:
        # Calculate the overlap area between two bounding boxes
        bbox1 = (self.left, self.top, self.right, self.bottom)
        bbox2 = (
            another_bbox.left,
            another_bbox.top,
            another_bbox.right,
            another_bbox.bottom,
        )
        overlap_left = max(bbox1[0], bbox2[0])
        overlap_top = max(bbox1[1], bbox2[1])
        overlap_right = min(bbox1[2], bbox2[2])
        overlap_bottom = min(bbox1[3], bbox2[3])
        bbox_intersect = Bbox(
            left=overlap_left,
            top=overlap_top,
            right=overlap_right,
            bottom=overlap_bottom,
        )
        return bbox_intersect.compute_area()

    def is_inside(self, another_bbox: "Bbox"):
        return (
            self.left >= another_bbox.left
            and self.right <= another_bbox.right
            and self.top >= another_bbox.top
            and self.bottom <= another_bbox.bottom
        )

    def is_mostly_inside(self, another_bbox: "Bbox", threshold: float = 0.85):
        area_inner = self.compute_area()
        overlap_area = self.compute_overlap_area(another_bbox)
        overlap_ratio = overlap_area / area_inner if area_inner > 0 else 1
        return overlap_ratio > threshold

    def is_overlapped(
        self,
        another_bbox: "Bbox",
        threshold1: float = 0.85,
        threshold2: float = 0.85,
        or_overlap: bool = False,
    ) -> bool:
        """
        Determine if the overlap of bbox1 with bbox2 exceeds
        threshold1 relative to bbox1's area and if the overlap of bbox2
        with bbox1 exceeds threshold2 relative to bbox2's area.

        Returns
        -------
        bool
            When or_overlap is True, returns True if either overlap exceeds
            its respective threshold.
            When or_overlap is False, returns True if both overlaps exceed
            their respective thresholds.
        """

        area1 = self.compute_area()
        area2 = another_bbox.compute_area()
        overlap_area = self.compute_overlap_area(another_bbox)

        # Check if the overlap exceeds the thresholds
        # relative to each bounding box's area
        overlap_ratio1 = overlap_area / area1 if area1 > 0 else 1
        overlap_ratio2 = overlap_area / area2 if area2 > 0 else 1

        if or_overlap:
            return overlap_ratio1 > threshold1 or overlap_ratio2 > threshold2

        return overlap_ratio1 > threshold1 and overlap_ratio2 > threshold2


def convert_to_global_bbox(
    outer_bbox: Optional[Bbox], inner_bbox_relative: Bbox
) -> Bbox:
    """
    Converts a bounding box (inner_bbox_relative) that is relative to
    another bounding box (outer_bbox) into coordinates that are relative
    to the entire image.

    Args:
        outer_bbox:
            The outer bounding box relative to the entire image.
        inner_bbox_relative:
            The inner bounding box relative to the outer bounding box.

    Returns:
        A new Bbox instance representing the inner bounding box
        relative to the entire image.
    """
    if outer_bbox is None:
        return inner_bbox_relative
    global_left = outer_bbox.left + inner_bbox_relative.left * (
        outer_bbox.right - outer_bbox.left
    )
    global_top = outer_bbox.top + inner_bbox_relative.top * (
        outer_bbox.bottom - outer_bbox.top
    )
    global_right = outer_bbox.left + inner_bbox_relative.right * (
        outer_bbox.right - outer_bbox.left
    )
    global_bottom = outer_bbox.top + inner_bbox_relative.bottom * (
        outer_bbox.bottom - outer_bbox.top
    )

    # Use the confidence from the inner bbox
    return Bbox(
        left=global_left,
        top=global_top,
        right=global_right,
        bottom=global_bottom,
        confidence=inner_bbox_relative.confidence,
    )


def convert_to_relative_bbox(outer_bbox: Bbox, inner_bbox_global: Bbox) -> Bbox:
    """
    Converts a bounding box (inner_bbox_global) that is relative to
    the entire image into coordinates that are relative to
    another bounding box (outer_bbox).

    Args:
        outer_bbox:
            The outer bounding box relative to the entire image.
        inner_bbox_global:
            The inner bounding box relative to the entire image.

    Returns:
        A new Bbox instance representing the inner bounding box
        relative to the outer bounding box.
    """
    relative_left = (inner_bbox_global.left - outer_bbox.left) / (
        outer_bbox.right - outer_bbox.left
    )
    relative_top = (inner_bbox_global.top - outer_bbox.top) / (
        outer_bbox.bottom - outer_bbox.top
    )
    relative_right = (inner_bbox_global.right - outer_bbox.left) / (
        outer_bbox.right - outer_bbox.left
    )
    relative_bottom = (inner_bbox_global.bottom - outer_bbox.top) / (
        outer_bbox.bottom - outer_bbox.top
    )

    # Use the confidence from the inner bbox
    return Bbox(
        left=relative_left,
        top=relative_top,
        right=relative_right,
        bottom=relative_bottom,
        confidence=inner_bbox_global.confidence,
    )


def crop_bbox(image: np.ndarray, bbox: Bbox):
    if not isinstance(bbox, Bbox):
        bbox = Bbox.model_validate(bbox)
    left = int(bbox.left * image.shape[1])
    top = int(bbox.top * image.shape[0])
    right = int(bbox.right * image.shape[1])
    bottom = int(bbox.bottom * image.shape[0])
    return image[top:bottom, left:right]


def union_bboxes(bboxes: list[Bbox]) -> Bbox:
    """
    Returns the smallest bounding box that contains all the given bounding boxes.
    """
    if not bboxes:
        return Bbox(0.0, 0.0, 0.0, 0.0)

    # Initialize min and max coordinates with the first bbox
    min_left = bboxes[0].left
    min_top = bboxes[0].top
    max_right = bboxes[0].right
    max_bottom = bboxes[0].bottom

    # Iterate through all bboxes to find the min and max coordinates
    for bbox in bboxes[1:]:
        min_left = min(min_left, bbox.left)
        min_top = min(min_top, bbox.top)
        max_right = max(max_right, bbox.right)
        max_bottom = max(max_bottom, bbox.bottom)

    # Return the bounding box that encompasses all bboxes
    return Bbox(left=min_left, top=min_top, right=max_right, bottom=max_bottom)


def remove_contained_bboxes(
    bboxes: list[Bbox], remove_inside: bool = True
) -> list[Bbox]:
    filtered_bboxes = []

    for index, bbox_to_check in enumerate(bboxes):

        # If remove_inside is True, remove bbox_to_check if it is inside any other bbox
        if remove_inside:
            # List to store whether each bbox is inside any other bbox
            is_contained = [
                bbox_to_check.is_inside(other_bbox) if i != index else False
                for i, other_bbox in enumerate(bboxes)
            ]
            if not any(is_contained):
                filtered_bboxes.append(bbox_to_check)
        # If remove_inside is False, remove bbox_to_check if it contains any other bbox
        else:
            contain_others = [
                other_bbox.is_inside(bbox_to_check) if i != index else False
                for i, other_bbox in enumerate(bboxes)
            ]
            if not any(contain_others):
                filtered_bboxes.append(bbox_to_check)

    return filtered_bboxes


def select_high_conf_bboxes(
    bboxes: list[Bbox], tolerent_diff: float = 0.05
) -> list[Bbox]:
    if len(bboxes) == 0:
        return bboxes
    max_confidence = np.max([bbox.confidence for bbox in bboxes])
    filtered_bboxes = []

    for bbox in bboxes:
        if max_confidence - bbox.confidence < tolerent_diff:
            filtered_bboxes.append(bbox)

    return filtered_bboxes
