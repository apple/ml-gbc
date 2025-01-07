# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import numpy as np
import supervision as sv

from .bbox import Bbox


def annotate_bboxes(
    image: np.ndarray,
    bboxes: list[Bbox],
    include_labels: bool = True,
):
    image_width, image_height = image.shape[1], image.shape[0]
    ref_length = (image_width + image_height) / 2
    thickness = int(ref_length / 100)
    text_thickness = int(thickness / 2)
    text_scale = ref_length / 500
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness, color=sv.Color.GREEN)
    label_annotator = sv.LabelAnnotator(
        text_thickness=text_thickness,
        text_position=sv.Position.CENTER,
        color=sv.Color.GREEN,
        text_scale=text_scale,
        text_padding=thickness,
        text_color=sv.Color.BLACK,
    )
    xyxy = []
    confidence = []
    for bbox in bboxes:
        if not isinstance(bbox, Bbox):
            bbox = Bbox.model_validate(bbox)
        xyxy.append(
            (
                bbox.left * image_width,
                bbox.top * image_height,
                bbox.right * image_width,
                bbox.bottom * image_height,
            )
        )
        confidence.append(bbox.confidence)
    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidence),
        class_id=np.zeros(len(bboxes), dtype=int),
    )
    annotated_image = image.copy()
    annotated_image = bounding_box_annotator.annotate(annotated_image, detections)
    if include_labels:
        annotated_image = label_annotator.annotate(
            annotated_image,
            detections,
            labels=[str(i) for i in range(1, len(detections) + 1)],
        )
    return annotated_image


def annotate_all_labels(image: np.ndarray, labeled_bboxes: list[tuple[str, Bbox]]):
    if len(labeled_bboxes) == 0:
        return image
    image_width, image_height = image.shape[1], image.shape[0]
    ref_length = min(image_width, image_height)
    thickness = int(ref_length / 100)
    text_thickness = int(thickness / 2)
    text_scale = ref_length / 800
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_thickness=text_thickness,
        text_scale=text_scale,
        text_padding=thickness,
        text_color=sv.Color.BLACK,
        text_position=sv.Position.TOP_RIGHT,
    )
    xyxy = []
    labels = []
    confidence = []
    class_id = []
    text_to_id = dict()
    current_id = 0
    for text, bbox in labeled_bboxes:
        if not isinstance(bbox, Bbox):
            bbox = Bbox.model_validate(bbox)
        xyxy.append(
            (
                bbox.left * image_width,
                bbox.top * image_height,
                bbox.right * image_width,
                bbox.bottom * image_height,
            )
        )
        confidence.append(bbox.confidence)
        if text not in text_to_id:
            text_to_id[text] = current_id
            current_id += 1
        labels.append(text)
        class_id.append(text_to_id[text])
    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidence),
        class_id=np.array(class_id),
    )
    annotated_image = image.copy()
    annotated_image = bounding_box_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(
        annotated_image, detections, labels=labels
    )
    return annotated_image
