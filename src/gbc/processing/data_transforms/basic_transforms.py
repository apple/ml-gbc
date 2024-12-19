# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
from collections.abc import Callable
from typing import Literal

from gbc.utils import get_gbc_logger
from gbc.data import GbcGraph, GbcGraphFull


def identity(x):
    return x


def append_string_to_filename(filename: str, append_text: str) -> str:
    filename, ext = os.path.splitext(filename)
    return filename + append_text + ext


def string_replace(string: str, old_str: str, new_str: str) -> str:
    return string.replace(old_str, new_str)


def to_gbc_graph_simple(gbc_graph_full: GbcGraphFull) -> GbcGraph:
    return gbc_graph_full.to_gbc_graph()


def to_gbc_graph_full(gbc_graph: GbcGraph) -> GbcGraphFull:
    if not isinstance(gbc_graph, GbcGraphFull):
        gbc_graph = GbcGraphFull.from_gbc_graph(gbc_graph)
    return gbc_graph


def modify_gbc_img_path(
    gbc_graph: GbcGraph, path_transform: Callable[[str], str]
) -> GbcGraph:
    if gbc_graph.img_path is not None:
        gbc_graph.img_path = path_transform(gbc_graph.img_path)
    return gbc_graph


def compute_caption_statistics(gbc_graph: GbcGraph) -> GbcGraphFull:
    if not isinstance(gbc_graph, GbcGraphFull):
        gbc_graph = GbcGraphFull.from_gbc_graph(gbc_graph)
    for vertex in gbc_graph.vertices:
        for caption in vertex.descs:
            caption.get_statistics()
    return gbc_graph


def gbc_graph_to_text_and_image(
    gbc_graph: GbcGraph,
    text_format: Literal["set", "set_with_bbox", "concat", "structured"] = "structured",
    graph_traversal_mode: Literal[
        "bfs", "dfs", "topological", "random"
    ] = "topological",
    caption_agg_mode_for_structured: Literal["first", "concat"] = "first",
    concat_separator: str = " ",
    read_image: bool = False,
    remove_repeated_suffix: bool = True,
) -> dict:
    if not isinstance(gbc_graph, GbcGraphFull):
        gbc_graph = GbcGraphFull.from_gbc_graph(gbc_graph)
    if text_format == "set":
        text = gbc_graph.get_captions(
            with_bbox=False,
            mode=graph_traversal_mode,
            remove_repeated_suffix=remove_repeated_suffix,
        )
    elif text_format == "set_with_bbox":
        text = gbc_graph.get_captions(
            with_bbox=True,
            mode=graph_traversal_mode,
            remove_repeated_suffix=remove_repeated_suffix,
        )
    elif text_format == "concat":
        text = gbc_graph.get_caption_concat(
            separator=concat_separator,
            mode=graph_traversal_mode,
            remove_repeated_suffix=remove_repeated_suffix,
        )
    elif text_format == "structured":
        text = gbc_graph.get_graph_text_repr(
            caption_agg_mode=caption_agg_mode_for_structured,
            graph_traversal_mode=graph_traversal_mode,
            concat_separator=concat_separator,
            remove_repeated_suffix=remove_repeated_suffix,
        )
    res = {
        "text": text,
        "image": None,
        "image_path": gbc_graph.img_path,
        "image_url": gbc_graph.img_url,
    }
    if read_image:
        image = gbc_graph.get_image()
        if image is None:
            logger = get_gbc_logger()
            logger.warning(
                "`read_image` is set to True, but image not found for GBC graph, "
                "so image field is set to None."
            )
        res["image"] = image
    return res


def basic_filter_and_extract(
    gbc_graph: GbcGraph,
    drop_composition_descendants: bool = False,
    drop_vertex_size_kwargs: dict[str, float] | None = None,
    drop_vertex_types: list[str] | None = None,
    drop_caption_types: list[str] | None = None,
    same_level_max_bbox_overlap_ratio: float | None = None,
    max_n_vertices: int | None = None,
    max_depth: int | None = None,
    subgraph_extraction_mode: Literal["bfs", "dfs"] = "bfs",
    subraph_edge_shuffling: bool = False,
    keep_in_edges: bool = True,
    keep_out_edges: bool = True,
) -> GbcGraphFull:
    if not isinstance(gbc_graph, GbcGraphFull):
        gbc_graph = GbcGraphFull.from_gbc_graph(gbc_graph)
    if drop_composition_descendants:
        gbc_graph = gbc_graph.drop_composition_descendants()
    if drop_vertex_size_kwargs is not None:
        gbc_graph = gbc_graph.drop_vertices_by_size(
            keep_in_edges=keep_in_edges,
            keep_out_edges=keep_out_edges,
            **drop_vertex_size_kwargs,
        )
    if drop_vertex_types is not None:
        gbc_graph = gbc_graph.drop_vertices_by_type(
            drop_vertex_types,
            keep_in_edges=keep_in_edges,
            keep_out_edges=keep_out_edges,
        )
    if drop_caption_types is not None:
        gbc_graph = gbc_graph.drop_captions_by_type(
            drop_caption_types,
            keep_in_edges=keep_in_edges,
            keep_out_edges=keep_out_edges,
        )
    if same_level_max_bbox_overlap_ratio is not None:
        gbc_graph = gbc_graph.drop_vertices_by_overlap_area(
            same_level_max_bbox_overlap_ratio,
            keep_in_edges=keep_in_edges,
            keep_out_edges=keep_out_edges,
        )
    if max_n_vertices is not None or max_depth is not None:
        gbc_graph = gbc_graph.get_subgraph(
            max_n_vertices=max_n_vertices,
            max_depth=max_depth,
            mode=subgraph_extraction_mode,
            edge_shuffling=subraph_edge_shuffling,
        )
    return gbc_graph
