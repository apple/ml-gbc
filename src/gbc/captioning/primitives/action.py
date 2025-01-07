# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
from typing import Optional, Type
from omegaconf import DictConfig
from PIL import Image

from gbc.utils import instantiate_class

from .io_unit import QueryResult
from .action_io import ActionInput, ActionOutput, NodeInfo


__all__ = [
    "Action",
    "ActionInputPair",
]


class Action(object):
    """
    Abstract base class for defining actions that can be queried.

    This class should be subclassed to implement specific actions. Subclasses must
    implement  the :meth:`~query`  method for single queries and can optionally
    override the :meth:`~batch_query` method for handling multiple queries
    more efficiently.
    """

    def query(
        self,
        action_input: ActionInput,
        queried_nodes: Optional[dict[str, list[NodeInfo]]] = None,
    ) -> ActionOutput:
        """
        Executes a query based on the provided action input

        This method performs the core action logic using the information
        in ``action_input``. It may also leverage previously queried nodes
        stored in ``queried_nodes``.

        Parameters
        ----------
        action_input:
            The input required to perform the action.
        queried_nodes:
            A dictionary containing information about previously queried nodes,
            as organized by image path.
            This is used to avoid redundant queries (via node merging) and is updated
            with new nodes discovered during this query.

        Returns
        -------
        ActionOutput
            The output of the action, containing the following:

            - actions_to_complete: Additional action input pairs for recursive query.
            - query_result: The result of the query, which may be ``None``.
            - image: The image used to perform the query, which may be ``None``.
        """
        pass

    def batch_query(
        self,
        action_inputs: list[ActionInput],
        queried_nodes: Optional[dict[str, list[NodeInfo]]],
    ) -> list[ActionOutput]:
        """
        Executes multiple queries in parallel.

        By default, this method sequentially calls :meth:`~query` for each input.
        Subclasses can override this method to provide a more efficient implementation.

        Parameters
        ----------
        action_inputs
            A list of inputs for performing the queries.
        queried_nodes
            A dictionary containing information about previously queried nodes,
            as organized by image path.
            This is used to avoid redundant queries (via node merging) and is updated
            with new nodes discovered during this query.

        Returns
        -------
        list[ActionOutput]
            A list of outputs, each corresponding to an action input.
        """
        return [self.query(input, queried_nodes) for input in action_inputs]

    @staticmethod
    def _add_to_queried_nodes(
        query_result: QueryResult,
        action_input: ActionInput,
        queried_nodes: Optional[dict[str, list[NodeInfo]]],
    ):
        """
        Adds the result of a query to the queried nodes dictionary.

        Parameters
        ----------
        query_result : QueryResult
            The result of the query to be added.
        action_input : ActionInput
            The input that produced the query result.
        queried_nodes : dict of str to list of NodeInfo, optional
            A dictionary containing information about previously queried nodes,
            as organized by image path.
            This is used to avoid redundant queries (via node merging) and is updated
            with new nodes discovered during this query.
        """
        if queried_nodes is None:
            return
        node_info = NodeInfo(
            action_input=action_input,
            query_result=query_result,
        )
        img_path = node_info.img_path
        if img_path in queried_nodes:
            queried_nodes[img_path].append(node_info)
        else:
            queried_nodes[img_path] = [node_info]


class ActionInputPair(object):
    """
    Encapsulates a pair of an :class:`~Action` class and its corresponding
    input for performing queries.

    This class allows for the storage and execution of queries.
    The storage is achieved through the :meth:`~model_dump`
    and :meth:`model_validate` methods.
    Precisely, the conversion from and to a dictionary relies on representing
    each class with their module and name.
    To execute the query, the action class should be instantiated with
    a configuration dictionary, passed as argument to the :meth:`~query` method.

    Attributes
    ----------
    action_class : Type[Action]
        The class of the action to be performed.
    action_input : ActionInput
        The input required for the action.
    """

    def __init__(self, action_class: Type[Action], action_input: ActionInput):
        self.action_class = action_class
        self.action_input = action_input

    def __repr__(self):
        return f"ActionInputPair({self.action_class!r}, {self.action_input!r})"

    def model_dump(self) -> dict:
        """
        Converts the ``ActionInputPair`` object to a dictionary format.

        Returns
        -------
        dict
            Dictionary representation of the ``ActionInputPair`` object.
        """
        return {
            "action_type": (
                self.action_class.__module__,
                self.action_class.__name__,
            ),
            "input_type": (
                self.action_input.__class__.__module__,
                self.action_input.__class__.__name__,
            ),
            "input_dict": self.action_input.model_dump(),
        }

    @classmethod
    def model_validate(cls, obj: dict) -> "ActionInputPair":
        """
        Validates and constructs an ``ActionInputPair`` object from a dictionary.

        Parameters
        ----------
        obj : dict
            Dictionary containing the action type and input type information.

        Returns
        -------
        ActionInputPair
            An instance of ``ActionInputPair`` constructed from the dictionary.

        Raises
        ------
        ValueError
            If the input is not a dictionary.
        """
        # Ensure obj is a dict
        if not isinstance(obj, dict):
            raise ValueError("Input should be a dictionary", cls)

        action_class = instantiate_class(*obj["action_type"])
        input_class = instantiate_class(*obj["input_type"])
        input = input_class.model_validate(obj["input_dict"])
        return cls(action_class=action_class, action_input=input)

    def query(
        self,
        config: DictConfig,
        queried_nodes: Optional[dict[str, list[NodeInfo]]] = None,
    ) -> ActionOutput:
        """
        Executes the query for the action using the provided configuration.

        It first instantiates :attr:`~action_class` with the provided configuration
        and then calls the ``query`` method of the action on the :attr:`~action_input`.

        Parameters
        ----------
        config
            Configuration for the querying pipeline.
            It would generally define how the actions should be instantiated.
        queried_nodes
            A dictionary containing information about previously queried nodes,
            as organized by image path.
            This is used to avoid redundant queries (via node merging) and is updated
            with new nodes discovered during this query.

        Returns
        -------
        ActionOutput
            The output of the action, containing the following:

            - actions_to_complete: Additional action input pairs for recursive query.
            - query_result: The result of the query, which may be ``None``.
            - image: The image used to perform the query, which may be ``None``.
        """
        action = self.action_class(config)
        return action.query(self.action_input, queried_nodes=queried_nodes)

    def save_image(self, image: Image.Image, base_save_dir: str):
        """
        Saves the associated image to a subdirectory within a specified base directory.

        This method takes the image to be saved and a base directory path.
        It constructs a subdirectory structure within ``base_save_dir``
        based on the following logic:

        - ``images``: A subdirectory is created under ``base_save_dir`` to store images.
        - Image Path (modified): The image path is used but adjusted by:

            - Splitting filename and extension using ``os.path.splitext``
            - Replacing path separators with hyphens (-)
        - Entity ID: The entity ID from the query node information is retrieved.
        - Action Class Name: The name of the current action class is retrieved.

        The final save path is constructed by joining these elements.

        Parameters
        ----------
        image
            The image to be saved.
        base_save_dir
            The base directory path where the image will be saved.
            Subdirectories will be created within this path.
        """
        img_path_adjusted = os.path.splitext(self.action_input.img_path)[0].replace(
            os.path.sep, "-"
        )
        entity_info = self.action_input.entity_info
        if isinstance(entity_info, list):
            entity_info = entity_info[0]
        save_name = (entity_info.entity_id + "_" + self.action_class.__name__).replace(
            os.path.sep, "-"
        )
        save_path = os.path.join(
            base_save_dir, "images", img_path_adjusted, save_name + ".jpg"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
