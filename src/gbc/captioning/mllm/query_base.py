# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from typing import Optional, Callable
from PIL import Image

from gbc.data.bbox import union_bboxes

from ..primitives import (
    Action,
    QueryResult,
    NodeInfo,
    ActionInput,
    ActionOutput,
    ActionInputPair,
    ActionInputWithEntities,
)


class BaseImageQuery(Action):
    """
    Abstract base class for image query actions.

    Its subclass should implement :meth:`~query_core` which returns
    :class:`~gbc.captioning.primitives.io_unit.QueryResult` and ``Image.Image``.
    From there, we deduce the detection action that should
    be performed on the image.

    Attributes
    ----------
    detection_action_class : Action
        The class representing the detection action to be used.
    suitable_for_detection_func : Callable | None
        Function to check if a string is suitable for detection.
        If not provided, a default function is used.
    """

    def __init__(
        self,
        detection_action_class: Action,
        suitable_for_detection_func: Optional[Callable] = None,
    ):
        self.detection_action_class = detection_action_class
        # Note that the default versions do not use any trained models
        if suitable_for_detection_func is None:
            from gbc.texts.text_helpers import suitable_for_detection

            suitable_for_detection_func = suitable_for_detection
        self.suitable_for_detection_func = suitable_for_detection_func

    def query_core(self, action_input: ActionInput) -> tuple[QueryResult, Image.Image]:
        """
        The core query logic.

        Parameters
        ----------
        action_input
            The input to the query.

        Returns
        -------
        query_result
            The result of the query containing descriptions, child entities,
            and raw output.
        image
            The image used as input to the query.
        """
        raise NotImplementedError

    def _parse_query_result_for_detection(
        self, action_input: ActionInput, query_result: QueryResult
    ) -> list[ActionInputPair]:
        # Only query entities that are suitable for detection
        entities = [
            entity
            for entity, _ in query_result.entities
            if self.suitable_for_detection_func(entity.text)
        ]

        actions_to_complete = []
        if len(entities) > 0:
            action_input_with_entities = ActionInputWithEntities(
                image=action_input.image,
                entity_info=action_input.entity_info,
                bbox=action_input.bbox,
                entities=entities,
            )
            detect_action_input_pair = ActionInputPair(
                action_class=self.detection_action_class,
                action_input=action_input_with_entities,
            )
            actions_to_complete.append(detect_action_input_pair)
        return actions_to_complete

    def query(
        self,
        action_input: ActionInput,
        queried_nodes: Optional[dict[str, list[NodeInfo]]] = None,
    ):
        # The core query logic
        query_result, image = self.query_core(action_input)
        self._add_to_queried_nodes(query_result, action_input, queried_nodes)
        actions_to_complete = self._parse_query_result_for_detection(
            action_input, query_result
        )
        return ActionOutput(actions_to_complete, query_result, image)


class BaseEntityQuery(Action):
    """
    Abstract base class for entity query actions.

    Its subclass should implement :meth:`~query_core` which returns
    :class:`~gbc.captioning.primitives.io_unit.QueryResult` and ``Image.Image``.

    This class is designed to avoid redundant queries by utilizing previously
    queried nodes provided in the input of the :meth:`~query` method. If a similar
    query has already been performed on the same content (determined by bounding
    box overlap with :attr:`~mask_overlap_threshold` and semantic similarity of
    the text with :attr:`~potential_same_object_func`), the query is skipped and
    the information from the two queries is merged into one.

    Attributes
    ----------
    detection_action_class : Action
        The class representing the detection action to be used.
    mask_overlap_threshold : float, default=0.85
        Threshold for determining bounding box overlap.
    suitable_for_detection_func : Callable | None
        Function to check if a string is suitable for detection.
        If not provided, a default function is used.
    potential_same_object_func : Callable | None
        Function to check if two strings refer to the same object.
        If not provided, a default function is used.
    """

    _parse_query_result_for_detection = BaseImageQuery._parse_query_result_for_detection

    def __init__(
        self,
        detection_action_class: Action,
        mask_overlap_threshold: float = 0.85,
        suitable_for_detection_func: Optional[Callable] = None,
        potential_same_object_func: Optional[Callable] = None,
    ):
        self.detection_action_class = detection_action_class
        self.mask_overlap_threshold = mask_overlap_threshold
        # Note that the default versions do not use any trained models
        if suitable_for_detection_func is None:
            from gbc.texts.text_helpers import suitable_for_detection

            suitable_for_detection_func = suitable_for_detection
        if potential_same_object_func is None:
            from gbc.texts.text_helpers import potential_same_object

            potential_same_object_func = potential_same_object
        self.suitable_for_detection_func = suitable_for_detection_func
        self.potential_same_object_func = potential_same_object_func

    def query_core(self, action_input: ActionInput) -> tuple[QueryResult, Image.Image]:
        """
        The core query logic.

        Parameters
        ----------
        action_input
            The input to the query.

        Returns
        -------
        query_result
            The result of the query containing descriptions, child entities,
            and raw output.
        image
            The image used as input to the query.
        """
        raise NotImplementedError

    def query(
        self,
        action_input: ActionInput,
        queried_nodes: Optional[dict[str, list[NodeInfo]]] = None,
    ):
        # Check if similar query has already been performed
        img_path = action_input.img_path
        if queried_nodes and img_path in queried_nodes:
            node_infos = queried_nodes[img_path]
            for node_info in node_infos:
                current_bbox = action_input.bbox
                stored_bbox = node_info.action_input.bbox
                if current_bbox is None or stored_bbox is None:
                    continue
                # In-place modification of node_info if it is a similar query
                if current_bbox.is_overlapped(
                    stored_bbox,
                    threshold1=self.mask_overlap_threshold,
                    threshold2=self.mask_overlap_threshold,
                ):
                    if not isinstance(node_info.action_input.entity_info, list):
                        entity_infos = [node_info.action_input.entity_info]
                    else:
                        entity_infos = node_info.action_input.entity_info
                    for entity_info in entity_infos:
                        current_text = action_input.entity_info.text
                        stored_text = entity_info.text
                        if current_text is None or stored_text is None:
                            continue
                        if self.potential_same_object_func(current_text, stored_text):
                            entity_infos.append(action_input.entity_info)
                            node_info.action_input.entity_info = entity_infos
                            # Update bbox to be union of the two
                            node_info.action_input.bbox = union_bboxes(
                                [
                                    node_info.action_input.bbox,
                                    action_input.bbox,
                                ],
                            )
                            return ActionOutput([], None, None)

        # The core query logic
        query_result, image = self.query_core(action_input)
        self._add_to_queried_nodes(query_result, action_input, queried_nodes)
        actions_to_complete = self._parse_query_result_for_detection(
            action_input, query_result
        )

        return ActionOutput(actions_to_complete, query_result, image)
