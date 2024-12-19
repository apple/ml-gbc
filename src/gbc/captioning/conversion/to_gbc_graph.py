# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
from collections import defaultdict
from PIL import Image
from typing import Optional

from gbc.utils import get_gbc_logger
from gbc.data import Bbox, Caption
from gbc.data.graph.gbc_graph import GbcEdge
from gbc.data.graph.gbc_graph_full import GbcVertexFull, GbcGraphFull

from .parse_node_infos import parse_node_info_list_per_image
from ..primitives import NodeInfo


def node_infos_to_gbc_graphs(
    node_infos: list[NodeInfo], mask_inside_threshold: float = 0.85
) -> list[GbcGraphFull]:
    """
    Convert a list of :class:`~gbc.captioning.primitives.action_io.NodeInfo`
    to a list of :class:`~gbc.data.graph.gbc_graph_full.GbcGraphFull`.

    Parameters
    ----------
    node_infos
        Node infos produced by the GBC captioning pipeline.
    mask_inside_threshold
        The threshold for determining whether a vertex region is contained
        in another vertex region. Used for ``sub_masks`` and ``super_masks``
        of :class:`~gbc.data.graph.gbc_graph_full.GbcVertexFull`.

    Returns
    -------
    list[GbcGraphFull]
        The GBC graphs obtained from the conversion.
    """
    img_to_node_infos = defaultdict(list)
    for node_info in node_infos:
        img_to_node_infos[node_info.img_path].append(node_info)

    graphs = []
    for img_path, node_infos in img_to_node_infos.items():
        graph_candidate = node_infos_to_gbc_graph(node_infos, mask_inside_threshold)
        if graph_candidate is not None:
            graphs.append(graph_candidate)

    return graphs


def node_infos_to_gbc_graph(
    node_infos: list[NodeInfo], mask_inside_threshold: float = 0.85
) -> Optional[GbcGraphFull]:
    """
    Convert a list of :class:`~gbc.captioning.primitives.action_io.NodeInfo`
    from **the same image** to a :class:`~gbc.data.graph.gbc_graph_full.GbcGraphFull`.

    Parameters
    ----------
    node_infos
        Node infos produced by the GBC captioning pipeline for a single image.
    mask_inside_threshold
        The threshold for determining whether a vertex region is contained
        in another vertex region. Used for ``sub_masks`` and ``super_masks``
        of :class:`~gbc.data.graph.gbc_graph_full.GbcVertexFull`.

    Returns
    -------
    Optional[GbcGraphFull]
        The GBC graph obtained from the conversion.
        Returns ``None`` if no graph is obtained.
    """

    vertices = []
    img_path = None
    id_to_node_info, entity_id_mapping = parse_node_info_list_per_image(node_infos)

    # This is caused by failure in llava querying
    # The response to image query is in the wrong format, giving rise to no descs
    # Note that we would still return the graph if the image query gives concise caption
    # but fails to provide detail captions
    if len(id_to_node_info) == 0:
        return None

    for entity_id, node_info in id_to_node_info.items():
        vertex = node_info_to_gbc_vertex(node_info, entity_id_mapping)
        vertices.append(vertex)

        if img_path is None:
            img_path = node_info.img_path
        assert (
            img_path == node_info.img_path
        ), f"Image path {img_path} does not match {node_info.img_path}"

    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            img_size = img.size
    else:
        img_size = None

    return GbcGraphFull(
        img_path=img_path,
        img_size=img_size,
        vertices=vertices,
        mask_inside_threshold=mask_inside_threshold,
    )


def node_info_to_gbc_vertex(
    node_info: NodeInfo, entity_id_mapping: dict[str, list[str]]
) -> GbcVertexFull:

    assert len(node_info.query_result.descs) > 0, "Empty query result is not allowed"

    # Use the first entity info
    if isinstance(node_info.action_input.entity_info, list):
        entity_info = node_info.action_input.entity_info[0]
    else:
        entity_info = node_info.action_input.entity_info

    # Define vertex_id
    vertex_id = entity_info.entity_id
    assert (
        vertex_id in entity_id_mapping
    ), f"Vertex id {vertex_id} not found in entity_id_mapping."
    assert entity_id_mapping[vertex_id] == [
        vertex_id
    ], f"First vertex id {vertex_id} should be mapped to itself."

    # Define bbox
    bbox = node_info.action_input.bbox
    if bbox is None:
        bbox = Bbox(left=0.0, top=0.0, right=1.0, bottom=1.0)

    # Define vertex label
    label = entity_info.label
    if label not in ["image", "entity", "composition", "relation"]:
        logger = get_gbc_logger()
        logger.warning(
            f"Unexpected label '{label}' encountered for "
            "action_input.entity_info.label. "
            "The node label will be set to 'entity'."
        )
        label = "entity"

    # Define vertex captions
    # The second part is only useful for relation description, but at this
    # point we already break relation query result into multiple node_infos
    descs = [
        Caption.from_desc_and_vertex_label(desc, label)
        for desc, _ in node_info.query_result.descs
    ]

    # Define outgoing edges
    out_edges = []
    for child_entity_info, _ in node_info.query_result.entities:
        # The child text should not be empty
        if child_entity_info.text.strip() == "":
            continue
        # If it is not found in the dictionary it means
        # there is no corresponding query result
        if child_entity_info.entity_id in entity_id_mapping:
            for target_entity_id in entity_id_mapping[child_entity_info.entity_id]:
                out_edges.append(
                    GbcEdge(
                        source=vertex_id,
                        text=child_entity_info.text,
                        target=target_entity_id,
                    )
                )

    # in_edges, sub_masks, and super_masks will be updated automatically
    # once the graph is constructed
    return GbcVertexFull(
        vertex_id=vertex_id,
        bbox=bbox,
        label=label,
        descs=descs,
        in_edges=[],
        out_edges=out_edges,
        sub_masks=[],
        super_masks=[],
    )
