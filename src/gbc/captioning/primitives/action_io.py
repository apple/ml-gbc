# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from PIL import Image
from typing import Any, NamedTuple, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel

import numpy as np

from gbc.utils import instantiate_class, ImageCache
from gbc.data.bbox import Bbox, crop_bbox
from .io_unit import EntityInfo, QueryResult

if TYPE_CHECKING:
    # Avoid circular import
    from .action import ActionInputPair


__all__ = [
    "ActionInput",
    "ActionInputWithEntities",
    "ActionInputWithBboxes",
    "ActionOutput",
    "NodeInfo",
    "get_action_input_from_img_path",
]


def get_action_input_from_img_path(img_path: str) -> "ActionInput":
    """
    Creates an :class:`~ActionInput` instance for a given image path.

    This function initializes the :class:`~ActionInput` with an
    :class:`~gbc.utils.ImageCache` object for the specified
    image path and default entity information suitable for Image queries.

    Parameters
    ----------
    img_path
        The path to the image file.

    Returns
    -------
    ActionInput
        An action input configured for the specified image path.
    """
    entity_info = EntityInfo(
        text=None,
        entity_id="",
        label="image",
    )
    action_input = ActionInput(
        image=ImageCache(img_path=img_path),
        entity_info=entity_info,
    )
    return action_input


class ActionInput(BaseModel):
    """
    The main class defining the input for and action.

    This class encapsulates the information needed to perform an action on an image.

    Attributes
    ----------
    image : ImageCache
        The image to perform the action on. It utilizes
        :class:`~gbc.utils.ImageCache` for lazy loading
        and to avoid repetition of loading of the same image.

    entity_info : EntityInfo | list[EntityInfo]
        Information about the root entity(ies) of the query. It indicates where
        the query originates from. It can be a list when multiple queries are
        merged due to overlapped content.
        This is not to be confused with ``entities`` in
        :class:`~ActionInputWithEntities`,
        which are entities that are relevant to the query results.

    bbox : Bbox | None
        The bounding box specifying the part of the image to perform the action on.
        Defaults to ``None``, meaning the entire image is used.
    """

    image: ImageCache
    entity_info: Union[EntityInfo, list[EntityInfo]]
    bbox: Optional[Bbox] = None

    # Cache the cropped image
    _image: Any = None

    @property
    def label(self):
        """
        Retrieves the label of the query based on entity information.

        Returns
        -------
        str
            The query label (``image``, ``entity``, ``relation``, or ``composition``).

        Raises
        ------
        ValueError
            If multiple entities in a list have different labels or
            if the entity label is unexpected.
        """
        if isinstance(self.entity_info, list):
            label = None
            for entity_info in self.entity_info:
                if label is None:
                    label = entity_info.label
                if label != entity_info.label:
                    raise ValueError(
                        f"Two different labels: {label} and {entity_info.label} "
                        "found for the same query"
                    )
                if label not in ["image", "entity", "relation", "composition"]:
                    raise ValueError(f"Unexpected label for query: {label}")
            return label
        return self.entity_info.label

    @property
    def img_path(self):
        """
        Returns the image path from the associated
        :class:`~gbc.utils.ImageCache` object.
        """
        return self.image.img_path

    @property
    def first_entity_id(self):
        """
        Retrieves the entity ID of the first entity in the :attr:`~entity_info`.
        """
        if isinstance(self.entity_info, list):
            return self.entity_info[0].entity_id
        return self.entity_info.entity_id

    def get_image(self, return_pil: bool = False) -> Union[Image.Image, np.ndarray]:
        """
        Get the image to perform the action on.

        Parameters
        ----------
        return_pil
            Whether to return the image as a PIL Image. Defaults to False.

        Returns
        -------
        Image | np.ndarray
            The image to perform the action on, with cropping if :attr:`~bbox`
            is not None.
        """
        if self._image is None:
            image = np.array(self.image.get_image())
            if self.bbox is not None:
                image = crop_bbox(image, self.bbox)
            self._image = image
        if return_pil:
            return Image.fromarray(self._image)
        return self._image

    def model_dump(self, *args, **kwargs) -> dict:
        """
        Converts the ``ActionInput`` to a dictionary, excluding the stored image.
        """
        # Remove the stored image during serialization to a dict
        output = super().model_dump(*args, **kwargs)
        output["image"]["image"] = None
        return output


class ActionInputWithEntities(ActionInput):
    """
    Subclass of :class:`~ActionInput` that includes a list of entities.

    Used for detection actions and relational queries.

    Attributes
    ----------
    entities : list[EntityInfo]
        List of entities relevant to the query result.
    """

    entities: list[EntityInfo]


class ActionInputWithBboxes(ActionInputWithEntities):
    """
    Subclass of :class:`~ActionInputWithEntities` that includes
    bounding boxes for each entity.

    Used for composition queries.

    Attributes
    ----------
    bboxes : list[Bbox]
        List of bounding boxes associated with the entities.
    """

    bboxes: list[Bbox]


class ActionOutput(NamedTuple):
    """
    A named tuple that encapsulates the output of an action query.

    Attributes
    ----------
    actions_to_complete: list[ActionInputPair]
        A list of action input pairs that represent
        the actions that need to be further completed.
    query_result: QueryResult | None
        The result of the query, if any.
        Importantly, `action_input` along with `query_result` together can be
        parsed to obtain the final GBC once all the actions are completed.
    image: Image | None
        Image that is used to perform the query.
    """

    actions_to_complete: list["ActionInputPair"]
    query_result: Optional[QueryResult]
    image: Optional[Image.Image]


class NodeInfo(BaseModel):
    """
    Defines the information needed from each node for the final query result.

    .. note::
       The main captioning pipeline stores the captioning artifacts as a
       list of ``NodeInfo``, which can be converted to
       :class:`~gbc.data.graph.gbc_graph_full.GbcGraphFull` once all the
       actions are completed.

    Attributes
    ----------
    action_input : ActionInput
        The input data for the action associated with this node.
    query_result : QueryResult
        The result of the query performed with the given action input.
    """

    # For alternative solutions for subclass resolution, see:
    # https://github.com/pydantic/pydantic/issues/7366

    action_input: ActionInput
    query_result: QueryResult

    @property
    def label(self) -> str:
        """Retrieves the label associated with the action input."""
        return self.action_input.label

    @property
    def img_path(self) -> str:
        """Retrieves the image path associated with the action input."""
        return self.action_input.img_path

    def model_dump(self) -> dict:
        """Converts the ``NodeInfo`` instance into a dictionary format."""
        return {
            "input_type": (
                self.action_input.__class__.__module__,
                self.action_input.__class__.__name__,
            ),
            "input_dict": self.action_input.model_dump(),
            "query_result": self.query_result.model_dump(),
        }

    @classmethod
    def model_validate(cls, obj: dict) -> "NodeInfo":
        """Validates and constructs a ``NodeInfo`` instance from a dictionary."""
        # Ensure obj is a dict
        if not isinstance(obj, dict):
            raise ValueError("Input should be a dictionary", cls)

        input_class = instantiate_class(*obj["input_type"])
        input = input_class.model_validate(obj["input_dict"])
        query_result = QueryResult.model_validate(obj["query_result"])
        return cls(action_input=input, query_result=query_result)
