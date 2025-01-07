# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import traceback
from typing import Optional

from gbc.utils import get_gbc_logger
from gbc.data.bbox import convert_to_global_bbox

from .detection import Detection, DetectionTextInput
from ..primitives import (
    EntityInfo,
    NodeInfo,
    ActionInput,
    ActionInputWithEntities,
    ActionOutput,
    Action,
    ActionInputPair,
)

from ..auto_actions import AutoEntityQuery


class DetectionAction(Action):
    """
    Action wrapper for object detection.

    The action takes an
    :class:`~gbc.captioning.primitives.action_io.ActionInputWithEntities`
    object as input, which contains a list of entities to detect.

    It then performs the following steps:

    1. Iterates through each entity and creates a
       :class:`~gbc.captioning.detection.detection.DetectionTextInput` object
       with appropriate parameters based on the entity label (single or multiple).
    2. Retrieves the image for processing.
    3. Calls the detection model to identify the entities in the image.
    4. In case of errors, logs a warning and returns an empty
       :class:`~gbc.captioning.primitives.action_io.ActionOutput`.
    5. Iterates through the detected entities and creates follow-up queries
       (of type :class:`~gbc.captioning.auto_actions.AutoEntityQuery`)
       for each detection bounding box.
    6. Returns an :class:`~gbc.captioning.primitives.action_io.ActionOutput`
       containing the list of queries and no
       :class:`~gbc.captioning.primitives.io_unit.QueryResult` or input image.

    At the heart of this class is the :attr:`~detection_model` attribute which
    implements the effective detection mechanism.
    It can be customized with different underlying detection models as needed.

    Attributes
    ----------
    detection_model : Detection | None , default=None
        The detection model to use.
        It uses :class:`~gbc.captioning.detection.grounding_dino.GroundingDinoDetection`
        when set to ``None``.
    score_threshold : float, default=0.05
        Minimum score threshold for filtering detections.
    nms_single_threshold : float, default=0.05
        Non-maximum suppression (NMS) threshold for entities labelled with "single".
    nms_multiple_threshold : float, default=0.2
        NMS threshold for entities labelled with "multiple".
    select_high_conf_tolerance : float | None, default=0.05
        Tolerance for selecting high-confidence detections.
        Only used for entities labelled with "single".
    topk : int | None, default=6
        Number of top detections to keep after NMS.
    min_abs_area : int | None, default=5000
        Minimum absolute area requirement for detections.
    max_rel_area : int | None, default=None
        Maximum relative area constraint for detections.

    Methods
    -------
    query(action_input, queried_nodes=None)
        Perform the detection action.
    """

    def __init__(
        self,
        detection_model: Optional[Detection] = None,
        score_threshold: float = 0.05,
        nms_single_threshold: float = 0.05,
        nms_multiple_threshold: float = 0.2,
        select_high_conf_tolerance: Optional[float] = 0.05,
        topk: Optional[int] = 6,
        min_abs_area: Optional[int] = 5000,
        max_rel_area: Optional[float] = None,
    ):
        if detection_model is None:
            # Use grounding_dino as default detection model
            from .grounding_dino import GroundingDinoDetection

            self.detection_model = GroundingDinoDetection()
        else:
            self.detection_model = detection_model
        self.score_threshold = score_threshold
        self.nms_single_threshold = nms_single_threshold
        self.nms_multiple_threshold = nms_multiple_threshold
        self.select_high_conf_tolerance = select_high_conf_tolerance
        self.topk = topk
        self.min_abs_area = min_abs_area
        self.max_rel_area = max_rel_area

    def query(
        self,
        action_input: ActionInputWithEntities,
        queried_nodes: Optional[dict[str, list[NodeInfo]]] = None,
    ) -> ActionOutput:
        entities = action_input.entities

        # We adjust bbox post-processing steps according to entity label
        detection_text_inputs = []
        for entity in entities:
            detection_text_input = DetectionTextInput(
                text=entity.text,
                score_threshold=self.score_threshold,
                min_abs_area=self.min_abs_area,
                max_rel_area=self.max_rel_area,
                nms_threshold=(
                    self.nms_single_threshold
                    if entity.label == "single"
                    else self.nms_multiple_threshold
                ),
                select_high_conf_tolerance=(
                    self.select_high_conf_tolerance
                    if entity.label == "single"
                    else None
                ),
                topk=self.topk,
            )
            detection_text_inputs.append(detection_text_input)
        image = action_input.get_image(return_pil=False)

        # When using yolo_world, the above fails with low probability
        # /io/opencv/modules/imgproc/src/resize.cpp:4065: error: (-215:Assertion failed) inv_scale_x > 0 in function 'resize'  # noqa
        # we also get
        # RuntimeError: nms_impl: implementation for device cuda:0 not found.
        # so we cannot simply ignore error, need a better solution here
        try:
            bbox_dict = self.detection_model.detect(image, detection_text_inputs)
        except Exception as e:
            logger = get_gbc_logger()
            logger.warning(f"Failed to detect: {e}")
            print("Traceback:")
            traceback.print_exc()
            raise e
            # return ActionOutput([], None, None)

        # Define entity queries according to the detected bbox
        query_list = []
        current_node_bbox = action_input.bbox
        for entity in entities:
            if entity.text not in bbox_dict:
                continue
            bboxes = bbox_dict[entity.text]
            for i, bbox in enumerate(bboxes):
                if len(bboxes) == 1:
                    entity_id = entity.entity_id
                else:
                    entity_id = f"{entity.entity_id}_{i}"
                entity_info = EntityInfo(
                    # singular_form should not be used here as the goal is to
                    # match the description
                    text=entity.text,
                    entity_id=entity_id,
                    label="entity",
                )
                bbox = convert_to_global_bbox(current_node_bbox, bbox)
                action_input = ActionInput(
                    image=action_input.image,
                    entity_info=entity_info,
                    bbox=bbox,
                )
                query = ActionInputPair(
                    action_class=AutoEntityQuery,
                    action_input=action_input,
                )
                query_list.append(query)
        # Note that we do not return image for detection action
        return ActionOutput(query_list, None, None)
