# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
from tqdm import tqdm
from copy import deepcopy
from typing import Any, Callable, Literal, Optional, Type, Union
from pydantic import BaseModel
from omegaconf import DictConfig
from collections import defaultdict

from gbc.utils import (
    get_gbc_logger,
    save_list_to_file,
    load_list_from_file,
    get_images_recursively,
)
from gbc.data.graph.gbc_graph_full import GbcGraphFull

from .get_relational_queries import get_relational_queries
from ..primitives import (
    NodeInfo,
    ActionInput,
    Action,
    ActionInputPair,
    get_action_input_from_img_path,
)
from ..auto_actions import AutoImageQuery
from ..conversion.to_gbc_graph import node_infos_to_gbc_graphs


def save_info(
    node_infos: list[NodeInfo],
    completed_actions: list[ActionInputPair],
    actions_to_complete: list[ActionInputPair],
    save_dir: str,
    format: Literal[".json", ".jsonl", ".parquet"] = ".jsonl",
):
    """
    Function used by :class:`~GbcPipeline` to save intermediate results.
    """
    save_list_to_file(node_infos, os.path.join(save_dir, "node_infos" + format))
    save_list_to_file(
        completed_actions, os.path.join(save_dir, "completed_actions" + format)
    )
    save_list_to_file(
        actions_to_complete, os.path.join(save_dir, "actions_to_complete" + format)
    )


def load_info(save_dir: str, format: Literal[".json", ".jsonl", ".parquet"] = ".jsonl"):
    """
    Function used by :class:`~GbcPipeline` to load intermediate results.
    """
    node_info_file = os.path.join(save_dir, "node_infos" + format)
    completed_action_file = os.path.join(save_dir, "completed_actions" + format)
    actions_to_complete_file = os.path.join(save_dir, "actions_to_complete" + format)

    if not os.path.exists(node_info_file):
        node_infos = []
    else:
        node_infos = load_list_from_file(node_info_file, NodeInfo)
    if not os.path.exists(completed_action_file):
        completed_actions = []
    else:
        completed_actions = load_list_from_file(completed_action_file, ActionInputPair)
    if not os.path.exists(actions_to_complete_file):
        actions_to_complete = []
    else:
        actions_to_complete = load_list_from_file(
            actions_to_complete_file, ActionInputPair
        )
    return node_infos, completed_actions, actions_to_complete


class GbcPipeline(BaseModel):
    """
    GBC Captioning Pipeline

    This class implements a pipeline for GBC captioning, allowing configuration
    and customization of various aspects of the captioning process.

    Attributes
    ----------
    captioning_cfg : DictConfig
        Configuration for the captioning pipeline, specifying the implementation
        of different actions. Optionally, one can specify the class's attributes
        in the ``pipeline`` section.

    save_frequency : int, default=10
        The frequency (in terms of completed actions) at which intermediate results
        are saved and the :attr:`~save_callback` is called.

    save_dir : str | None, default=None
        Directory to save intermediate results. If not specified, intermediate results
        may only be saved using :attr:`~save_callback`.

    artifact_format : Literal[".json", ".jsonl", ".parquet"], default=".jsonl"
        The format of the intermediate results.

    save_callback : Callable | None, default=None
        An optional callback function for saving intermediate results.
        The callback must accept three arguments:
        ``node_infos``, ``completed_actions``, and ``actions_to_complete``.

        .. note::
           Setting
           :attr:`~save_callback` to ``partial(save_info, save_dir=save_dir)``
           and  :attr:`~save_dir` to ``None`` and
           is equivalent to setting
           :attr:`~save_dir` to ``save_dir`` and :attr:`~save_callback` to ``None``.

    save_images : bool, default=False
        Whether to save images used for each query.

    batch_query : bool, default=False
        Whether to perform batch queries.

    batch_size : int, default=32
        The batch size for batch queries.

    include_entity_query: bool, default=True
        Whether to include entity queries.

    include_composition_query: bool, default=True
        Whether to include composition queries.

    include_relation_query: bool, default=True
        Whether to include relation queries.

    mask_inside_threshold : float, default=0.85
        The threshold for determining if a mask is inside another mask.
        It is used for conversion to GBC graphs at the end.
        Precisely, it affects ``sub_masks`` and ``super_masks`` of each vertex.
    """

    captioning_cfg: Any

    # Arguments for saving intermediate results
    artifact_format: Literal[".json", ".jsonl", ".parquet"] = ".jsonl"
    save_frequency: int = 10
    save_dir: Optional[str] = None
    save_callback: Optional[Callable] = None
    save_images: bool = False

    # Arguments for batch processing
    batch_query: bool = False
    batch_size: int = 32

    # Arguments for determining which types of queries to include
    include_entity_query: bool = True
    include_composition_query: bool = True
    include_relation_query: bool = True

    # Arguments to determine if a mask is in another mask
    mask_inside_threshold: float = 0.85

    @classmethod
    def from_config(cls, config: DictConfig, **kwargs):
        """
        Instantiates the GBC pipeline from a configuration.

        For any attributes of the class, the values are determined in the following
        order of priority:

        1. kwargs
        2. ``pipeline_config`` section in config
        3. default values

        Parameters
        ----------
        config
            Configuration for the pipeline.
        **kwargs :
            Additional keyword arguments used to initialize the pipeline.

        Returns
        -------
        GbcPipeline
            The instantiated pipeline.

        Raises
        ------
        ValueError
            If there are unexpected keys in ``kwargs`` or ``config.pipeline_config``.
        """
        # Extract pipeline config if available
        pipeline_config = config.get("pipeline_config", {})

        # Allowed keys in the pipeline configuration
        allowed_keys = set(cls.__fields__.keys())

        # Check for unexpected arguments
        unexpected_keys = set(kwargs.keys()) - allowed_keys
        if unexpected_keys:
            raise ValueError(f"Unexpected keys in kwargs: {unexpected_keys}")

        unexpected_keys = set(pipeline_config.keys()) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                f"Unexpected keys in config.pipeline_config: {unexpected_keys}"
            )

        # Combine kwargs and pipeline_config with priority:
        # kwargs > pipeline_config > default values
        init_args = {}
        for key in allowed_keys:
            if key == "captioning_cfg":
                init_args[key] = config  # Directly set captioning_cfg to config
            else:
                if key in kwargs:
                    init_args[key] = kwargs[key]
                elif key in pipeline_config:
                    init_args[key] = pipeline_config[key]
                else:
                    init_args[key] = cls.__fields__[key].default

        return cls(**init_args)

    def model_post_init(self, context):
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.save_dir is None and self.save_images:
            raise ValueError("save_dir must be specified if save_images is True")

    def __call__(self, *args, **kwargs):
        return self.run_gbc_captioning(*args, **kwargs)

    def run_gbc_captioning(
        self,
        img_files_or_folders: Union[str, list[str]],
        *,
        attempt_resume: bool = True,
        return_raw_results: bool = False,
    ) -> Union[
        list[GbcGraphFull],
        tuple[list[NodeInfo], list[ActionInputPair], list[ActionInputPair]],
    ]:
        """
        Run the full GBC captioning on the provided images or folders according
        to the underlying configuration.

        Parameters
        ----------
        img_files_or_folders
            The image files or folders to be processed.
        attempt_resume
            Whether to attempt resuming from the save directory.

            - If ``True``, we try to load the previous results from the save directory,
              resume captioning from where we left off, and append the new results
              to the old ones.
            - If ``False``, we ignore all the previous results, perform captioning
              from scratch, and overwrite any existing artifacts in the save directory.
        return_raw_results
            Whether to return the GBC graphs or raw results.

            - If ``True``, return the triple of (``node_infos``,
              ``completed_actions``, ``actions_to_complete``).
            - If ``False``, return the GBC graphs parsed from the
              resulting node infos.

        Returns
        -------
        list[GbcGraphFull] | \
        tuple[list[NodeInfo], list[ActionInputPair], list[ActionInputPair]]

            Either the GBC graphs or the intermediate results, depending on
            the value of ``return_raw_results``.
        """
        if self.save_dir is not None and attempt_resume:
            node_infos, completed_actions, actions_to_complete = self.resume_captioning(
                recursive=True,
                return_raw_results=True,
            )
            assert len(actions_to_complete) == 0
        else:
            node_infos = []
            completed_actions = []

        # Composition and relation query can only be performed if
        # there are entity queries
        has_second_stage = self.include_composition_query or self.include_relation_query
        if has_second_stage and not self.include_entity_query:
            logger = get_gbc_logger()
            logger.warning(
                "Composition and relation queries would not be performed "
                "if there were no entity queries."
            )
            has_second_stage = False

        results = self.run_image_entity_captioning(
            img_files_or_folders,
            node_infos=node_infos,
            completed_actions=completed_actions,
            tqdm_desc=self._get_image_entity_captioning_desc(),
            return_raw_results=return_raw_results or has_second_stage,
        )
        if has_second_stage:
            node_infos, completed_actions, actions_to_complete = results
            assert len(actions_to_complete) == 0
            results = self.run_relational_captioning(
                node_infos,
                completed_actions=completed_actions,
                tqdm_desc=self._get_relational_captioning_desc(),
                return_raw_results=return_raw_results,
            )
        return results

    def _get_image_entity_captioning_desc(self):
        if self.include_entity_query:
            return "Perform image and entity captioning"
        return "Perform image captioning"

    def _get_relational_captioning_desc(self):
        if self.include_relation_query and self.include_composition_query:
            return "Perform relation and composition captioning"
        elif self.include_relation_query and not self.include_composition_query:
            return "Perform relation captioning"
        elif self.include_composition_query and not self.include_relation_query:
            return "Perform composition captioning"
        return None

    def run_image_entity_captioning(
        self,
        img_files_or_folders: Union[str, list[str]],
        *,
        node_infos: Optional[list[NodeInfo]] = None,
        completed_actions: Optional[list[ActionInputPair]] = None,
        tqdm_desc: Optional[str] = None,
        return_raw_results: bool = False,
    ) -> Union[
        list[GbcGraphFull],
        tuple[list[NodeInfo], list[ActionInputPair], list[ActionInputPair]],
    ]:
        """
        Run image and entity captioning on the provided images or folders according
        to the underlying configuration.

        Parameters
        ----------
        img_files_or_folders
            The image files or folders to be processed.
        node_infos
            The node infos that have been obtained so far.
            We check against these to avoid reprocessing the same images.
            It also serves to make sure the saved information takes into account
            all the actions that have been completed so far.
        completed_actions
            The actions that have been completed so far.
            It serves to make sure the saved information takes into account
            all the actions that have been completed so far.
        tqdm_desc
            The description of the progress bar.
        return_raw_results
            Whether to return the GBC graphs or raw results.

            - If ``True``, return the triple of (``node_infos``,
              ``completed_actions``, ``actions_to_complete``).
            - If ``False``, return the GBC graphs parsed from the
              resulting node infos.

        Returns
        -------
        list[GbcGraphFull] | \
        tuple[list[NodeInfo], list[ActionInputPair], list[ActionInputPair]]

            Either the GBC graphs or the intermediate results, depending on
            the value of ``return_raw_results``.
        """
        if isinstance(img_files_or_folders, str):
            img_files_or_folders = [img_files_or_folders]
        img_files = []
        for img_folder in img_files_or_folders:
            if os.path.isdir(img_folder):
                img_files_tmp = get_images_recursively(img_folder)
                img_files.extend(img_files_tmp)
            else:
                img_files.append(img_folder)

        assert len(img_files) == len(set(img_files)), "all image file must be unique"

        node_infos = node_infos or []
        completed_actions = completed_actions or []

        for node_info in node_infos:
            if node_info.action_input.img_path in img_files:
                logger = get_gbc_logger()
                logger.info(
                    f"{node_info.img_path} queried results "
                    "found in save directory, skipping..."
                )
                img_files.remove(node_info.img_path)

        action_input_pairs = [
            ActionInputPair(
                action_class=AutoImageQuery,
                action_input=get_action_input_from_img_path(img_file),
            )
            for img_file in img_files
        ]
        return self.run_queries(
            action_input_pairs,
            node_infos=node_infos,
            completed_actions=completed_actions,
            # We should run queries recursively to include entity queries
            recursive=self.include_entity_query,
            init_queried_nodes_from_node_infos=False,
            tqdm_desc=tqdm_desc,
            return_raw_results=return_raw_results,
        )

    def run_relational_captioning(
        self,
        node_infos: list[NodeInfo],
        *,
        completed_actions: list[Action] = None,
        tqdm_desc: Optional[str] = None,
        return_raw_results: bool = False,
    ):
        """
        Run relation and composition captioning from the provided node infos
        according to the underlying configuration.

        Parameters
        ----------
        node_infos
            The node infos from which we infer the queries to be performed.
        completed_actions
            The actions that have been completed so far.
            It serves to make sure the saved information takes into account
            all the actions that have been completed so far.
        tqdm_desc
            The description of the progress bar.
        return_raw_results
            Whether to return the GBC graphs or raw results.

            - If ``True``, return the triple of (``node_infos``,
              ``completed_actions``, ``actions_to_complete``).
            - If ``False``, return the GBC graphs parsed from the
              resulting node infos.

        Returns
        -------
        list[GbcGraphFull] | \
        tuple[list[NodeInfo], list[ActionInputPair], list[ActionInputPair]]

            Either the GBC graphs or the intermediate results, depending on
            the value of ``return_raw_results``.
        """

        completed_actions = completed_actions or []

        if not (self.include_composition_query or self.include_relation_query):
            return node_infos, completed_actions, []

        composition_queries = []
        relation_queries = []

        gbc_graphs = node_infos_to_gbc_graphs(node_infos)
        for graph in gbc_graphs:
            composition_queries_graph, relation_queries_graph = get_relational_queries(
                graph
            )
            composition_queries.extend(composition_queries_graph)
            relation_queries.extend(relation_queries_graph)

        action_input_pairs = []
        if self.include_composition_query:
            action_input_pairs.extend(composition_queries)
        if self.include_relation_query:
            action_input_pairs.extend(relation_queries)

        return self.run_queries(
            action_input_pairs,
            node_infos=node_infos,
            completed_actions=completed_actions,
            recursive=False,  # No need to query recursively for relational queries
            init_queried_nodes_from_node_infos=False,
            tqdm_desc=tqdm_desc,
            return_raw_results=return_raw_results,
        )

    def resume_captioning(
        self,
        *,
        recursive: bool = True,
        return_raw_results: bool = False,
    ):
        """
        Resume captioning from :attr:`~save_dir`.

        Parameters
        ----------
        recursive
            Whether to complete the actions recursively.
            If set to ``False``, the actions that are derived from the
            query results will not be completed, and we do not try
            to infer additional relational queries that are to be performed.
        return_raw_results
            Whether to return the GBC graphs or raw results.

            - If ``True``, return the triple of (``node_infos``,
              ``completed_actions``, ``actions_to_complete``).
            - If ``False``, return the GBC graphs parsed from the
              resulting node infos.

        Returns
        -------
        list[GbcGraphFull] | \
        tuple[list[NodeInfo], list[ActionInputPair], list[ActionInputPair]]

            Either the GBC graphs or the intermediate results, depending on
            the value of ``return_raw_results``.
        """
        node_infos, completed_actions, actions_to_complete = load_info(
            self.save_dir,
            self.artifact_format,
        )
        if len(actions_to_complete) == 0:
            logger = get_gbc_logger()
            logger.info("No actions to resume")
            if return_raw_results:
                return node_infos, completed_actions, actions_to_complete
            return node_infos_to_gbc_graphs(node_infos)

        logger = get_gbc_logger()
        logger.info(
            f"{len(actions_to_complete)} actions remain to be completed from "
            "save directory, resuming..."
        )

        has_second_stage = self.include_composition_query or self.include_relation_query
        # Only perform second stage if we query recursively
        if has_second_stage and not recursive:
            logger = get_gbc_logger()
            logger.warning(
                "When resuming, no additional relational queries would be added"
                "if recursive=False."
            )
            has_second_stage = False

        results = self.run_queries(
            actions_to_complete,
            node_infos=node_infos,
            completed_actions=completed_actions,
            recursive=recursive,
            init_queried_nodes_from_node_infos=True,
            tqdm_desc=f"Resume captioning from {self.save_dir}",
            return_raw_results=return_raw_results or has_second_stage,
        )
        if has_second_stage:
            node_infos, completed_actions, actions_to_complete = results
            assert len(actions_to_complete) == 0
            results = self.run_relational_captioning(
                node_infos,
                completed_actions=completed_actions,
                tqdm_desc=f"Perform relational queries from {self.save_dir}",
                return_raw_results=return_raw_results,
            )
        return results

    def run_queries(
        self,
        action_input_pairs: list[ActionInputPair],
        *,
        node_infos: Optional[list[NodeInfo]] = None,
        completed_actions: Optional[list[ActionInputPair]] = None,
        recursive: bool = True,
        init_queried_nodes_from_node_infos: bool = True,
        tqdm_desc: Optional[str] = None,
        return_raw_results: bool = False,
    ):
        """
        Run queries on the given action input pairs.

        Parameters
        ----------
        action_input_pairs
            The list of action input pairs to query.
        node_infos
            The node infos that have been obtained so far.
            It serves to make sure the saved information takes into account
            all the actions that have been completed so far.
            Moreover, when ``init_queried_nodes_from_node_infos`` is ``True``,
            we initialize the ``queried_nodes`` passed to
            :meth:`Action.query() <gbc.captioning.primitives.action.Action.query>`
            from the node infos.
            This is important when resuming from unfinished queries.
        completed_actions
            The actions that have been completed so far.
            It serves to make sure the saved information takes into account
            all the actions that have been completed so far.
        recursive
            Whether to complete the actions recursively.
            If set to ``False``, the actions that are derived from the
            query results will not be completed.
        init_queried_nodes_from_node_infos
            Whether to initialize the queried nodes from the node infos.
        tqdm_desc
            The tqdm description.
        return_raw_results
            Whether to return the GBC graphs or raw results.

            - If ``True``, return the triple of (``node_infos``,
              ``completed_actions``, ``actions_to_complete``).
            - If ``False``, return the GBC graphs parsed from the
              resulting node infos.

        Returns
        -------
        list[GbcGraphFull] | \
        tuple[list[NodeInfo], list[ActionInputPair], list[ActionInputPair]]

            Either the GBC graphs or the intermediate results, depending on
            the value of ``return_raw_results``.
        """
        query_func = self._batch_run_queries if self.batch_query else self._run_queries
        node_infos, completed_actions, actions_to_complete = query_func(
            action_input_pairs,
            node_infos=node_infos,
            completed_actions=completed_actions,
            recursive=recursive,
            init_queried_nodes_from_node_infos=init_queried_nodes_from_node_infos,
            tqdm_desc=tqdm_desc,
        )
        if return_raw_results:
            return node_infos, completed_actions, actions_to_complete
        gbc_graphs = node_infos_to_gbc_graphs(
            node_infos, mask_inside_threshold=self.mask_inside_threshold
        )
        return gbc_graphs

    def _run_queries(
        self,
        action_input_pairs: list[ActionInputPair],
        *,
        node_infos: Optional[list[NodeInfo]] = None,
        completed_actions: Optional[list[ActionInputPair]] = None,
        recursive: bool = True,
        init_queried_nodes_from_node_infos: bool = True,
        tqdm_desc: Optional[str] = None,
    ) -> tuple[list[NodeInfo], list[ActionInputPair], list[ActionInputPair]]:
        logger = get_gbc_logger()
        logger.info("Running queries...")

        # Avoid in-place modification
        actions_to_complete = action_input_pairs
        actions_to_complete_update = []
        node_infos = [] if node_infos is None else deepcopy(node_infos)
        completed_actions = (
            [] if completed_actions is None else deepcopy(completed_actions)
        )

        # Initialize queried node dictionary
        node_info_dict = defaultdict(list)
        if init_queried_nodes_from_node_infos:
            for node_info in node_infos:
                img_path = node_info.img_path
                node_info_dict[img_path].append(node_info)

        # Define tqdm pbar and create save directory
        tqdm_total = len(action_input_pairs) if not recursive else None
        pbar = tqdm(total=tqdm_total, desc=tqdm_desc)

        # Main captioning pipeline
        while len(actions_to_complete) > 0:
            for idx, action_input_pair in enumerate(actions_to_complete):
                actions, result, image = action_input_pair.query(
                    self.captioning_cfg, queried_nodes=node_info_dict
                )
                actions_to_complete_update.extend(actions)
                completed_actions.append(action_input_pair)
                if result is not None:
                    node_info = NodeInfo(
                        action_input=action_input_pair.action_input,
                        query_result=result,
                    )
                    node_infos.append(node_info)
                if self.save_images and image is not None and self.save_dir is not None:
                    action_input_pair.save_image(image, self.save_dir)
                if len(completed_actions) % self.save_frequency == 0:
                    if self.save_dir is not None:
                        save_info(
                            node_infos,
                            completed_actions,
                            actions_to_complete[idx + 1 :] + actions_to_complete_update,
                            self.save_dir,
                            self.artifact_format,
                        )
                    if self.save_callback is not None:
                        self.save_callback(
                            node_infos,
                            completed_actions,
                            actions_to_complete[idx + 1 :] + actions_to_complete_update,
                        )
                pbar.update(1)
            if recursive:
                actions_to_complete = actions_to_complete_update
                actions_to_complete_update = []
            else:
                actions_to_complete = []
        pbar.close()
        if not recursive:
            actions_to_complete = actions_to_complete_update
        else:
            assert len(actions_to_complete) == 0
        if self.save_dir is not None:
            save_info(
                node_infos,
                completed_actions,
                actions_to_complete,
                self.save_dir,
                self.artifact_format,
            )
        if self.save_callback is not None:
            self.save_callback(node_infos, completed_actions, actions_to_complete)
        return node_infos, completed_actions, actions_to_complete

    @staticmethod
    def _action_input_pairs_list2dict(
        action_input_pairs: list[ActionInputPair],
    ) -> dict[Type[Action], list[ActionInput]]:
        actions_to_complete_dict = defaultdict(list)
        for action_input_pair in action_input_pairs:
            actions_to_complete_dict[action_input_pair.action_class].append(
                action_input_pair.action_input
            )
        return actions_to_complete_dict

    @staticmethod
    def _action_input_pairs_dict2list(
        actions_to_complete_dict: dict[Type[Action], list[ActionInput]],
    ) -> list[ActionInputPair]:
        action_input_pairs = []
        for action_class, action_inputs in actions_to_complete_dict.items():
            for action_input in action_inputs:
                action_input_pairs.append(ActionInputPair(action_class, action_input))
        return action_input_pairs

    def _batch_run_queries(
        self,
        action_input_pairs: list[ActionInputPair],
        *,
        node_infos: Optional[list[NodeInfo]] = None,
        completed_actions: Optional[list[ActionInputPair]] = None,
        recursive: bool = True,
        init_queried_nodes_from_node_infos: bool = True,
        tqdm_desc: Optional[str] = None,
    ) -> tuple[list[NodeInfo], list[ActionInputPair], list[ActionInputPair]]:
        logger = get_gbc_logger()
        logger.info("Running queries in batch mode...")

        # Sort the action_input_pairs by action class for batch processing
        actions_to_complete_dict = self._action_input_pairs_list2dict(
            action_input_pairs
        )
        actions_to_complete_remained = []

        node_infos = [] if node_infos is None else deepcopy(node_infos)
        completed_actions = (
            [] if completed_actions is None else deepcopy(completed_actions)
        )
        n_currently_saved = 0

        # Initialize queried node dictionary
        node_info_dict = defaultdict(list)
        if init_queried_nodes_from_node_infos:
            for node_info in node_infos:
                img_path = node_info.img_path
                node_info_dict[img_path].append(node_info)

        # No tqdm_total here as the effective batch size could vary
        # (we need to perform action of the same type)
        pbar = tqdm(desc=tqdm_desc)

        # Main captioning pipeline
        while any(actions_to_complete_dict.values()):
            # Find the key with the list that has the most elements
            action_with_most_inputs = max(
                actions_to_complete_dict, key=lambda k: len(actions_to_complete_dict[k])
            )
            # Get the list with the most elements
            largest_list = actions_to_complete_dict[action_with_most_inputs]
            # Determine the number of elements to take (up to batch_size)
            num_inputs_to_take = min(self.batch_size, len(largest_list))
            # Get the elements from the beginning of the list
            inputs_to_process = largest_list[:num_inputs_to_take]
            # Remove the elements from the list
            actions_to_complete_dict[action_with_most_inputs] = largest_list[
                num_inputs_to_take:
            ]

            # If the list is now empty, we remove the key from the dictionary
            if not actions_to_complete_dict[action_with_most_inputs]:
                del actions_to_complete_dict[action_with_most_inputs]

            outputs = action_with_most_inputs(self.captioning_cfg).batch_query(
                inputs_to_process, queried_nodes=node_info_dict
            )
            action_input_pairs = [
                ActionInputPair(
                    action_class=action_with_most_inputs, action_input=action_input
                )
                for action_input in inputs_to_process
            ]
            for output, action_input_pair in zip(outputs, action_input_pairs):
                for to_complete in output.actions_to_complete:
                    if recursive:
                        actions_to_complete_dict[to_complete.action_class].append(
                            to_complete.action_input
                        )
                    else:
                        actions_to_complete_remained.append(to_complete)
                completed_actions.append(action_input_pair)
                if output.query_result is not None:
                    node_info = NodeInfo(
                        action_input=action_input_pair.action_input,
                        query_result=output.query_result,
                    )
                    node_infos.append(node_info)
                if (
                    self.save_images
                    and output.image is not None
                    and self.save_dir is not None
                ):
                    action_input_pair.save_image(output.image, self.save_dir)
            if len(completed_actions) >= n_currently_saved + self.save_frequency:
                actions_to_complete = (
                    self._action_input_pairs_dict2list(actions_to_complete_dict)
                    + actions_to_complete_remained
                )
                if self.save_dir is not None:
                    save_info(
                        node_infos,
                        completed_actions,
                        actions_to_complete,
                        self.save_dir,
                        self.artifact_format,
                    )
                if self.save_callback is not None:
                    self.save_callback(
                        node_infos, completed_actions, actions_to_complete
                    )
                n_currently_saved = len(completed_actions)
            pbar.update(1)
        pbar.close()
        actions_to_complete = actions_to_complete_remained
        if self.save_dir is not None:
            save_info(
                node_infos,
                completed_actions,
                actions_to_complete,
                self.save_dir,
                self.artifact_format,
            )
        if self.save_callback is not None:
            self.save_callback(node_infos, completed_actions, actions_to_complete)
        return node_infos, completed_actions, actions_to_complete
