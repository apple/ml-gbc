# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
from typing import Optional, Callable, Union

import numpy as np
from PIL import Image

from gbc.texts import plural_to_singular
from gbc.data import Description
from gbc.data.bbox import convert_to_relative_bbox
from gbc.data.bbox.annotate import annotate_bboxes

from .parse_mllm_output import (
    parse_image,
    parse_relation,
    parse_entity,
    parse_composition,
)
from .query_base import BaseImageQuery, BaseEntityQuery
from .composition_mst import MSTNode

from ..primitives import (
    Action,
    QueryResult,
    ActionInput,
    ActionOutput,
    ActionInputWithEntities,
    ActionInputWithBboxes,
)
from ..auto_actions import (
    AutoDetectionActionFromImage,
    AutoDetectionActionFromEntity,
)


class MllmQueryPrototype(Action):
    """
    Base class for some prototypes of MLLM query actions.

    This class provides a foundational structure for building query actions that
    interact with an MLLM. It handles loading system and query messages from specified
    files, manages the underlying MLLM model, and facilitates passing additional
    keyword arguments for queries.

    Subclasses are expected to implement their own :meth:`~load_model` and
    :meth:`~query_prelim` methods to define how the model is loaded and how queries

    Attributes
    ----------
    query_file : str
        Path to the query message file. Its content is loaded into
        ``self.query_message``. This message can be treated as a Python f-string,
        allowing placeholders that can be filled dynamically with ``filled_in_query``
        during each call.
    system_file : str, optional
        Path to the system message file. Its content is loaded into
        ``self.system_message``.
        If not provided, ``self.system_message`` is set to None.
    query_model : Any
        The underlying MLLM model instance used for querying.
        Subclasses should define the :meth:`~load_model` method to load and return
        an appropriate model from the provided keyword arguments.
    query_kwargs : dict, optional
        Additional keyword arguments passed to the MLLM query function.
    """

    def __init__(
        self,
        query_file: str,
        system_file: Optional[str] = None,
        model_kwargs: dict = {},
        query_kwargs: dict = {},
    ):
        if not os.path.exists(query_file):
            raise ValueError(f"Query file {query_file} does not exist")
        with open(query_file, "r") as f:
            self.query_message = f.read()
        if system_file is not None:
            if not os.path.exists(system_file):
                raise ValueError(f"System file {system_file} does not exist")
            with open(system_file, "r") as f:
                self.system_message = f.read()
        else:
            self.system_message = None
        self.query_file = query_file
        self.system_file = system_file
        self.query_model = self.load_model(**model_kwargs)
        self.query_kwargs = query_kwargs

    def load_model(self, **kwargs):
        """
        Load the underlying MLLM model.

        This method is intended to be overridden by subclasses to provide a
        concrete implementation for loading a multimodal large language model.
        The returned model should be capable of processing both textual and
        visual inputs as required.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments that may be required for model initialization.

        Returns
        -------
        Any
            An instance of the MLLM model, ready to be queried.
        """

        raise NotImplementedError

    def query_prelim(
        self,
        image: Image.Image,
        filled_in_query: Optional[Union[str, list[str], tuple[str]]] = None,
    ) -> QueryResult:
        """
        Execute a preliminary MLLM query with the provided image and
        formatted query message.

        This method is responsible for interfacing with the loaded MLLM model,
        applying the system and query messages, and returning the query result.
        It allows filling placeholders within the query message with the provided
        ``filled_in_query`` content, which can be a single string or a
        collection of strings.

        Parameters
        ----------
        image
            The image to query using the MLLM.
        filled_in_query
            Content to fill in the placeholders of the query message.

            - If a string, it's assumed to replace the **single** placeholder
              in the query message.
            - If a list or tuple, its length must match the number of
              placeholders. Each element replaces a corresponding placeholder.

        Returns
        -------
        QueryResult
            The result of the query containing descriptions, child entities,
            and raw output.
        """
        raise NotImplementedError

    def __repr__(self):
        repr = (
            f"{self.__class__.__name__}"
            f"(system_file={self.system_file!r},"
            f" query_file={self.query_file!r})"
        )
        return repr


class ImageQuery(BaseImageQuery, MllmQueryPrototype):
    """
    Image query implementation using an MLLM.

    It takes the vanilla :class:`~gbc.captioning.primitives.action_io.ActionInput`
    as input, which provides an image context. The query is performed using the
    loaded MLLM and the resulting output is parsed accordingly.

    For more details, please refer to the documentation of the parent classes.

    Attributes
    ----------
    system_file : str
        Path to the system prompt file.
    query_file : str
        Path to the user prompt file.
    query_model : Any
        The MLLM model instance for performing queries.
    suitable_for_detection_func : Callable | None
        A function to determine if a string is suitable for detection.
    parse_output_func : Callable
        A function to parse the model's output into a structured form.
    """

    def __init__(
        self,
        query_file: str,
        system_file: Optional[str] = None,
        model_kwargs: dict = {},
        suitable_for_detection_func: Optional[Callable] = None,
        parse_output_func: Optional[Callable] = None,
        query_kwargs: dict = {},
    ):
        MllmQueryPrototype.__init__(
            self,
            query_file,
            system_file,
            model_kwargs=model_kwargs,
            query_kwargs=query_kwargs,
        )
        BaseImageQuery.__init__(
            self,
            detection_action_class=AutoDetectionActionFromImage,
            suitable_for_detection_func=suitable_for_detection_func,
        )
        self.parse_output_func = parse_output_func or parse_image

    def query_core(self, action_input: ActionInput) -> tuple[QueryResult, Image.Image]:
        image = action_input.get_image(return_pil=True)
        query_output = self.query_prelim(image, None)
        query_result = self.parse_output_func(query_output, action_input)
        return query_result, image


class EntityQuery(BaseEntityQuery, MllmQueryPrototype):
    """
    Entity query implementation using an MLLM.

    It takes the vanilla :class:`~gbc.captioning.primitives.action_io.ActionInput`
    as input.
    For more details please refer to the documentation of the parent classes.

    Attributes
    ----------
    system_file : str
        Path to the system prompt file.
    query_file : str
        Path to the user prompt file.
    query_model : Any
        The MLLM model instance for performing queries.
    suitable_for_detection_func : Callable | None
        Function to check if a string is suitable for detection.
    potential_same_object_func : Callable | None
        Function to check if two strings are potential same object.
    parse_output_func : Callable | None
        A function to parse the model's output into a structured form.
    """

    def __init__(
        self,
        query_file: str,
        system_file: Optional[str] = None,
        model_kwargs: dict = {},
        mask_overlap_threshold: float = 0.85,
        suitable_for_detection_func: Optional[Callable] = None,
        potential_same_object_func: Optional[Callable] = None,
        parse_output_func: Optional[Callable] = None,
        query_kwargs: dict = {},
    ):
        MllmQueryPrototype.__init__(
            self,
            query_file,
            system_file,
            model_kwargs=model_kwargs,
            query_kwargs={
                "max_tokens": query_kwargs.pop("max_tokens", 512),
                **query_kwargs,
            },
        )
        BaseEntityQuery.__init__(
            self,
            detection_action_class=AutoDetectionActionFromEntity,
            mask_overlap_threshold=mask_overlap_threshold,
            suitable_for_detection_func=suitable_for_detection_func,
            potential_same_object_func=potential_same_object_func,
        )
        self.parse_output_func = parse_output_func or parse_entity

    def query_core(self, action_input: ActionInput) -> tuple[QueryResult, Image.Image]:
        image = action_input.get_image(return_pil=True)
        # For entity query, entity_info is never a list in action_input
        text = action_input.entity_info.text
        if text is not None:
            text = plural_to_singular(text)
            text = (text, text)
        query_output = self.query_prelim(image, text)
        query_result = self.parse_output_func(query_output, action_input)
        return query_result, image


class RelationQuery(MllmQueryPrototype):
    """
    Relation query implementation using an MLLM.

    It takes :class:`~gbc.captioning.primitives.action_io.ActionInputWithEntities`
    as input, where ``entities`` indicate the entities to be related.

    Attributes
    ----------
    system_file : str
        Path to the system prompt file.
    query_file : str
        Path to the user prompt file.
    query_model : Any
        The MLLM model instance for performing queries.
    parse_output_func : Callable | None
        A function to parse the model's output into a structured form.
    """

    def __init__(
        self,
        query_file: str,
        system_file: Optional[str] = None,
        model_kwargs: dict = {},
        parse_output_func: Optional[Callable] = None,
        query_kwargs: dict = {},
    ):
        MllmQueryPrototype.__init__(
            self,
            query_file,
            system_file,
            model_kwargs=model_kwargs,
            query_kwargs=query_kwargs,
        )
        self.parse_output_func = parse_output_func or parse_relation

    def query(
        self, action_input: ActionInputWithEntities, queried_nodes=None
    ) -> ActionOutput:

        entities = action_input.entities
        if len(entities) < 2:
            return ActionOutput([], None, None)
        to_relate = f"[{entities[0].text}]"
        for i in range(1, len(entities)):
            to_relate += f", [{entities[i].text}]"
        filled_in_query = np.random.randint(2, len(entities) + 1), to_relate

        image = action_input.get_image(return_pil=True)
        query_output = self.query_prelim(image, filled_in_query)
        query_result = self.parse_output_func(query_output, action_input)
        self._add_to_queried_nodes(query_result, action_input, queried_nodes)
        return ActionOutput([], query_result, image)


class CompositionQuery(MllmQueryPrototype):
    """
    Composition query implementation using an MLLM.

    It takes :class:`~gbc.captioning.primitives.action_io.ActionInputWithBboxes`
    as input, where ``bboxes`` and ``entities`` indicate the bounding boxes to
    be composed and their content.

    .. note::

       This composition query makes use of images annotated with bounding boxes
       and hard-coded hints generated with the help of
       :class:`~gbc.captioning.primitives.composition_mst.MSTNode`.

    Attributes
    ----------
    system_file : str
        Path to the system prompt file.
    query_file : str
        Path to the user prompt file.
    query_model : Any
        The MLLM model instance for performing queries.
    parse_output_func : Callable | None
        A function to parse the model's output into a structured form.
    """

    def __init__(
        self,
        query_file: str,
        system_file: Optional[str] = None,
        model_kwargs: dict = {},
        parse_output_func: Optional[Callable] = None,
        query_kwargs: dict = {},
    ):
        MllmQueryPrototype.__init__(
            self,
            query_file,
            system_file,
            model_kwargs=model_kwargs,
            query_kwargs=query_kwargs,
        )
        self.parse_output_func = parse_output_func or parse_composition

    def query(self, action_input: ActionInputWithBboxes, queried_nodes=None):
        image = action_input.get_image(return_pil=False)
        # Make the bounding boxes to be with respect to the cropped image
        bboxes = [
            convert_to_relative_bbox(action_input.bbox, bbox)
            for bbox in action_input.bboxes
        ]

        mst_node = MSTNode.build_mst(
            action_input.entities, abs_bboxes=action_input.bboxes, rel_bboxes=bboxes
        )
        descriptions, node_texts = mst_node.get_tree_descriptions()
        descriptions_with_dash = ["- " + description for description in descriptions]
        additional_description = "\n".join(descriptions_with_dash)

        annotated_image = annotate_bboxes(image, bboxes)
        annotated_image = Image.fromarray(annotated_image)
        entity_info = action_input.entity_info
        # By design it is possible to put multiple entity infos in one query
        # (see the code of `RelationalQueryManager`)
        if isinstance(entity_info, list):
            entity_info = entity_info[0]
        text = entity_info.text
        text = (
            f"{len(action_input.bboxes)} {text}",
            text,
            ", ".join(node_texts),
            additional_description,
        )
        query_output = self.query_prelim(annotated_image, text)
        query_result = self.parse_output_func(query_output, action_input)
        query_result.descs.extend(
            [
                (
                    Description(text=description, label="hardcode"),
                    [entity_info.entity_id],
                )
                for description in descriptions
            ]
        )
        self._add_to_queried_nodes(query_result, action_input, queried_nodes)
        return ActionOutput([], query_result, annotated_image)
