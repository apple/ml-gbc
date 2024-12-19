# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""
This module implements the functional interface of the GBC pipeline.
The arguments of :class:`~gbc.captioning.pipeline.pipeline.GbcPipeline` can be
passed as keyword arguments to the functions defined in this module.
"""

from typing import Optional, Union
from omegaconf import DictConfig

from .pipeline import GbcPipeline
from ..primitives import Action, ActionInputPair, NodeInfo


def run_gbc_captioning(
    img_files_or_folders: Union[str, list[str]],
    captioning_cfg: DictConfig,
    *,
    attempt_resume: bool = True,
    return_raw_results: bool = False,
    **kwargs,
):
    """
    Functional wrapper for
    :meth:`gbc.captioning.pipeline.pipeline.GbcPipeline.run_gbc_captioning`.
    """
    gbc_pipeline = GbcPipeline.from_config(captioning_cfg, **kwargs)
    return gbc_pipeline.run_gbc_captioning(
        img_files_or_folders,
        attempt_resume=attempt_resume,
        return_raw_results=return_raw_results,
    )


def run_image_entity_captioning(
    img_files_or_folders: Union[str, list[str]],
    captioning_cfg: DictConfig,
    *,
    node_infos: list[NodeInfo] = None,
    completed_actions: list[ActionInputPair] = None,
    tqdm_desc: Optional[str] = None,
    return_raw_results: bool = False,
    **kwargs,
):
    """
    Functional wrapper for
    :meth:`gbc.captioning.pipeline.pipeline.GbcPipeline.run_image_entity_captioning`.
    """
    gbc_pipeline = GbcPipeline.from_config(captioning_cfg, **kwargs)
    return gbc_pipeline.run_image_entity_captioning(
        img_files_or_folders,
        node_infos=node_infos,
        completed_actions=completed_actions,
        tqdm_desc=tqdm_desc,
        return_raw_results=return_raw_results,
    )


def run_relational_captioning(
    node_infos: list[NodeInfo],
    captioning_cfg: DictConfig,
    *,
    completed_actions: list[Action] = None,
    tqdm_desc: Optional[str] = None,
    return_raw_results: bool = False,
    **kwargs,
):
    """
    Functional wrapper for
    :meth:`gbc.captioning.pipeline.pipeline.GbcPipeline.run_relational_captioning`.
    """
    gbc_pipeline = GbcPipeline.from_config(captioning_cfg, **kwargs)
    return gbc_pipeline.run_relational_captioning(
        node_infos=node_infos,
        completed_actions=completed_actions,
        tqdm_desc=tqdm_desc,
        return_raw_results=return_raw_results,
    )


def resume_captioning(
    save_dir: str,
    captioning_cfg: DictConfig,
    *,
    recursive: bool = True,
    return_raw_results: bool = False,
    **kwargs,
):
    """
    Functional wrapper for
    :meth:`gbc.captioning.pipeline.pipeline.GbcPipeline.resume_captioning`.
    """
    gbc_pipeline = GbcPipeline.from_config(captioning_cfg, save_dir=save_dir, **kwargs)
    return gbc_pipeline.resume_captioning(
        recursive=recursive,
        return_raw_results=return_raw_results,
    )


def run_queries(
    action_input_pairs: list[ActionInputPair],
    captioning_cfg: DictConfig,
    *,
    node_infos: Optional[list[NodeInfo]] = None,
    completed_actions: Optional[list[ActionInputPair]] = None,
    recursive: bool = True,
    init_queried_nodes_from_node_infos: bool = True,
    tqdm_desc: Optional[str] = None,
    return_raw_results: bool = False,
    **kwargs,
):
    """
    Functional wrapper for
    :meth:`gbc.captioning.pipeline.pipeline.GbcPipeline.run_queries`.
    """
    gbc_pipeline = GbcPipeline.from_config(captioning_cfg, **kwargs)
    return gbc_pipeline.run_queries(
        action_input_pairs,
        node_infos=node_infos,
        completed_actions=completed_actions,
        recursive=recursive,
        init_queried_nodes_from_node_infos=init_queried_nodes_from_node_infos,
        tqdm_desc=tqdm_desc,
        return_raw_results=return_raw_results,
    )
