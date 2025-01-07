# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import re

from gbc.data.bbox import Bbox
from gbc.data.caption import Description
from gbc.data.graph import GbcVertex, GbcEdge, GbcGraph


def parse_gbc_graph(result: str, prompt: str, verbose: bool = False):

    node_parser: re.Pattern = re.compile(
        r"Node #(\d+)(.*)\n"
        r"type: (.+)\n"
        r"is_leave: (True|False)\n"
        r"desc: (.+)\n"
        r"parents:((?:\s*#\d+\((?:(?!#).)*\))*)\n"
        r"bbox: (.*)"
    )
    parents_parser: re.Pattern = re.compile(
        r"\s*#(\d)+\(((?:(?!#).)*)\: ((?:(?!#).)*)\)"
    )
    bbox_parser: re.Pattern = re.compile(r"((?:(?!\:).)*): ([\d\.]+)(?:, )*")

    vertices = {}
    for match in node_parser.finditer(result):
        id, vertex_id, vertex_label, is_leave, description, parents, bbox = (
            match.groups()
        )
        id = id.strip()
        vertex_id = vertex_id.strip()
        vertex_label = vertex_label.strip()
        is_leave = {"True": True, "False": False}[is_leave.strip()]
        description = description.strip()
        parents = parents_parser.findall(parents)
        try:
            bbox = bbox_parser.findall(bbox)
            bbox = {k: float(v) for k, v in bbox}
        except Exception:
            bbox = {}
        if bbox == {}:
            continue
        vertices[vertex_id] = GbcVertex(
            vertex_id=vertex_id,
            label=vertex_label,
            bbox=Bbox(**bbox),
            descs=[Description(text=description, label="short")],
            in_edges=[],
            out_edges=[],
        )
        in_edges = []
        for pid, parent_vid, text in parents:
            if parent_vid in vertices:
                edge = GbcEdge(
                    source=parent_vid, target=vertex_id, text=text, label="short"
                )
                vertices[parent_vid].out_edges.append(edge)
                in_edges.append(edge)
        vertices[vertex_id].in_edges = in_edges
        if verbose:
            print("=" * 60)
            print(
                f"{vertex_id} | {vertex_label} | {is_leave}\n"
                f"{parents} | {bbox}\n"
                f"{description}"
            )

    gbc_graph = GbcGraph(
        vertices=list(vertices.values()),
        short_caption=prompt,
    )
    return gbc_graph
