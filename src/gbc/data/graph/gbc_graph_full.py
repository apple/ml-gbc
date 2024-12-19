# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from typing import Optional, Literal, Union
from typing_extensions import Self
from collections import defaultdict
from copy import deepcopy
from pydantic import model_validator

import numpy as np

from gbc.texts import remove_repeated_suffix as remove_repeated_suffix_func
from .gbc_graph import GbcVertex, GbcEdge, GbcGraph
from ..caption import Description, Caption


class GbcVertexFull(GbcVertex):

    descs: list[Caption]
    # The vertices whose bboxes are mostly contained in the bbox of this vertex
    sub_masks: list[str] = []
    # The vertices whose bboxes mostly contain the bbox of this vertex
    super_masks: list[str] = []

    def to_gbc_vertex(self) -> GbcVertex:
        gbc_vertex_dict = self.model_dump()
        gbc_vertex_dict["descs"] = [
            Description(text=desc.text, label=desc.label) for desc in self.descs
        ]
        del gbc_vertex_dict["sub_masks"]
        del gbc_vertex_dict["super_masks"]
        return GbcVertex.model_validate(gbc_vertex_dict)

    @classmethod
    def from_gbc_vertex(cls, vertex: GbcVertex) -> "GbcVertexFull":
        descs = [
            Caption.from_desc_and_vertex_label(desc, vertex.label)
            for desc in vertex.descs
        ]
        # Note that we cannot get the value of `sub_masks` and `super_masks`
        # at this point
        return cls(
            vertex_id=vertex.vertex_id,
            bbox=vertex.bbox,
            label=vertex.label,
            descs=descs,
            in_edges=vertex.in_edges,
            out_edges=vertex.out_edges,
        )

    def model_post_init(self, context):
        for caption in self.descs:
            caption.full_label = caption.get_full_label_from_vertex_label(self.label)

    @property
    def is_leaf(self) -> bool:
        return len(self.out_edges) == 0

    def update_edges_with_in_edges(self, vertex_dict: dict[str, "GbcVertexFull"]):
        """
        Update the parents of the vertices
        """
        in_edges_update = []
        for edge in self.in_edges:
            # In edge can be removed when we extract subgraph
            if edge.source not in vertex_dict:
                continue
            parent = vertex_dict[edge.source]
            if edge not in parent.out_edges:
                parent.out_edges.append(edge)
            in_edges_update.append(edge)
        self.in_edges = in_edges_update

    def update_edges_with_out_edges(self, vertex_dict: dict[str, "GbcVertexFull"]):
        """
        Update the children of the vertices
        """
        out_edges_update = []
        for edge in self.out_edges:
            # Out edge can be removed when some vertices are filtered
            if edge.target not in vertex_dict:
                continue
            child = vertex_dict[edge.target]
            if edge not in child.in_edges:
                child.in_edges.append(edge)
            out_edges_update.append(edge)
        self.out_edges = out_edges_update

    def update_mask_relations(
        self, vertices: list["GbcVertexFull"], inside_threshold: float = 0.85
    ):
        """
        Update the sub_masks and super_masks of the vertices
        """
        self.sub_masks = []
        self.super_masks = []
        for vertex_alt in vertices:
            if vertex_alt.vertex_id == self.vertex_id:
                continue
            self.update_mask_relations_single_vertex(vertex_alt, inside_threshold)

    def update_mask_relations_single_vertex(
        self, vertex_alt: "GbcVertexFull", inside_threshold: float = 0.85
    ):
        """
        Update the sub_masks and super_masks of the vertices
        """
        if self.bbox.is_mostly_inside(vertex_alt.bbox, inside_threshold):
            if self.vertex_id not in vertex_alt.sub_masks:
                vertex_alt.sub_masks.append(self.vertex_id)
            if vertex_alt.vertex_id not in self.super_masks:
                self.super_masks.append(vertex_alt.vertex_id)
        if vertex_alt.bbox.is_mostly_inside(self.bbox, inside_threshold):
            if vertex_alt.vertex_id not in self.sub_masks:
                self.sub_masks.append(vertex_alt.vertex_id)
            if self.vertex_id not in vertex_alt.super_masks:
                vertex_alt.super_masks.append(self.vertex_id)


class GbcGraphFull(GbcGraph):

    vertices: list[GbcVertexFull]

    # Width and height of the image
    img_size: Optional[tuple[int, int]] = None

    # The threshold to determine if a vertex region is contained
    # in another vertex region
    mask_inside_threshold: float = 0.85

    # Cached values
    _vertex_dict: Optional[dict[str, GbcVertexFull]] = None
    _bfs_order: Optional[list[str]] = None
    _depth: Optional[int] = None
    _node_depths: Optional[dict[str, int]] = None
    _depth_to_nodes: Optional[dict[int, list[str]]] = None
    _dfs_order: Optional[list[str]] = None
    _topological_order: Optional[list[str]] = None

    def to_gbc_graph(self) -> GbcGraph:
        return GbcGraph(
            vertices=[vertex.to_gbc_vertex() for vertex in self.vertices],
            img_url=self.img_url,
            img_path=self.img_path,
            original_caption=self.original_caption,
            short_caption=self.short_caption,
            detail_caption=self.detail_caption,
        )

    @classmethod
    def from_gbc_graph(cls, gbc_graph: GbcGraph, mask_inside_threshold: float = 0.85):
        return cls(
            vertices=[
                GbcVertexFull.from_gbc_vertex(vertex) for vertex in gbc_graph.vertices
            ],
            img_url=gbc_graph.img_url,
            img_path=gbc_graph.img_path,
            original_caption=gbc_graph.original_caption,
            short_caption=gbc_graph.short_caption,
            detail_caption=gbc_graph.detail_caption,
            mask_inside_threshold=mask_inside_threshold,
        )

    def get_image_size(self, img_root_dir: str = ""):
        image = self.get_image(img_root_dir)
        if image is not None:
            self.img_size = image.size
        return self.img_size

    """
    Properties
    """

    @property
    def roots(self) -> list[GbcVertex]:
        image_roots = []
        additional_roots = []
        for vertex in self.vertices:
            if vertex.label == "image":
                if len(vertex.in_edges) > 0:
                    raise ValueError("Global vertex should not have in edges")
                image_roots.append(vertex)
            elif len(vertex.in_edges) == 0:
                additional_roots.append(vertex)
        return image_roots + additional_roots

    @property
    def leaves(self) -> list[GbcVertex]:
        leaves = []
        for vertex in self.vertices:
            if len(vertex.out_edges) == 0:
                leaves.append(vertex)
        return leaves

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return sum([len(vertex.out_edges) for vertex in self.vertices])

    @property
    def n_captions(self) -> int:
        return sum([len(vertex.descs) for vertex in self.vertices])

    @property
    def n_roots(self) -> int:
        return len(self.roots)

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    @property
    def n_pixels(self) -> Optional[int]:
        if self.img_size is None:
            return None
        return self.img_size[0] * self.img_size[1]

    @property
    def bfs_order(self) -> list[str]:
        if self._bfs_order is None:
            self._compute_bfs_and_depth()
        return self._bfs_order

    @property
    def depth(self) -> int:
        if self._depth is None:
            self._compute_bfs_and_depth()
        return self._depth

    @property
    def node_depths(self) -> dict[str, int]:
        if self._node_depths is None:
            self._compute_bfs_and_depth()
        return self._node_depths

    @property
    def depth_to_nodes(self) -> dict[int, list[str]]:
        if self._depth_to_nodes is None:
            self._compute_bfs_and_depth()
        return self._depth_to_nodes

    @property
    def dfs_order(self) -> list[str]:
        if self._dfs_order is None:
            self._compute_dfs_and_topological()
        return self._dfs_order

    @property
    def topological_order(self) -> list[str]:
        if self._topological_order is None:
            self._compute_dfs_and_topological()
        return self._topological_order

    @property
    def vertex_dict(self) -> dict[str, GbcVertexFull]:
        if self._vertex_dict is None:
            self._vertex_dict = {vertex.vertex_id: vertex for vertex in self.vertices}
        return self._vertex_dict

    """
    Computation of cached values
    """

    @model_validator(mode="after")
    def reset_cache(self) -> Self:
        self._depth = None
        self._bfs_order = None
        self._dfs_order = None
        self._topological_order = None
        self._vertex_dict = None
        self.update_edges_and_masks()
        return self

    def update_edges_and_masks(self):
        """
        In-place update of the edges and masks of vertices
        """
        # We sort the vertex list to get ordered ids
        # in `vertex.super_masks` and `vertex.sub_masks`
        vertices_list = sorted(self.vertices, key=lambda x: x.vertex_id)

        # Reset the masks
        for vertex in vertices_list:
            vertex.sub_masks = []
            vertex.super_masks = []

        for k, vertex in enumerate(vertices_list):
            vertex.update_edges_with_out_edges(self.vertex_dict)
            vertex.update_edges_with_in_edges(self.vertex_dict)
            vertex.update_mask_relations(
                vertices_list, inside_threshold=self.mask_inside_threshold
            )

    def _compute_dfs_and_topological(self):
        dfs_order = []
        dfs_end_order = []
        visited = set()
        current_vertices = [(vertex, False) for vertex in self.roots]

        # Note that a vertex can be added multiple times if it is not a tree
        while len(current_vertices) > 0:
            # Using a stack
            vertex, processed = current_vertices.pop()
            if processed:
                dfs_end_order.append(vertex.vertex_id)
                continue
            if vertex.vertex_id in visited:
                continue
            visited.add(vertex.vertex_id)
            dfs_order.append(vertex.vertex_id)
            # record entering point
            current_vertices.append((vertex, True))
            # Reversing edge order here to get the correct order
            for edge in reversed(vertex.out_edges):
                if edge.target not in visited:
                    current_vertices.append((self.vertex_dict[edge.target], False))

        self._dfs_order = dfs_order
        self._topological_order = list(reversed(dfs_end_order))

    def _compute_bfs_and_depth(self):
        depth = -1
        bfs_order = []
        node_depths = {}
        depth_to_nodes = defaultdict(list)
        visited = set()

        current_vertices = self.roots.copy()
        vertices_update = []
        while len(current_vertices) > 0:
            depth += 1
            for vertex in current_vertices:
                if vertex.vertex_id in visited:
                    continue
                visited.add(vertex.vertex_id)
                bfs_order.append(vertex.vertex_id)
                node_depths[vertex.vertex_id] = depth
                depth_to_nodes[depth].append(vertex.vertex_id)
                for edge in vertex.out_edges:
                    if edge.target not in visited:
                        vertices_update.append(self.vertex_dict[edge.target])
            current_vertices = vertices_update
            vertices_update = []

        self._depth = depth
        self._bfs_order = bfs_order
        self._node_depths = node_depths
        self._depth_to_nodes = depth_to_nodes

    """
    Text representation
    """

    def get_order(
        self,
        mode: Literal["bfs", "dfs", "topological", "random"] = "bfs",
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()
        if mode == "bfs":
            order = self.bfs_order
        elif mode == "dfs":
            order = self.dfs_order
        elif mode == "topological":
            order = self.topological_order
        elif mode == "random":
            order = rng.permutation(self.bfs_order)
        else:
            raise ValueError(f"Unknown mode {mode}")
        return order

    def get_captions(
        self,
        with_bbox: bool = False,
        mode: Literal["bfs", "dfs", "topological", "random"] = "bfs",
        remove_repeated_suffix: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[list[str] | list[tuple[str, tuple[float, float, float, float]]]]:
        order = self.get_order(mode, rng)
        captions = []
        for vertex_id in order:
            vertex = self.vertex_dict[vertex_id]
            for desc in vertex.descs:
                text = desc.text
                if remove_repeated_suffix:
                    text = remove_repeated_suffix_func(text)
                if with_bbox:
                    captions.append((text, vertex.bbox.to_xyxy()))
                else:
                    captions.append(text)
        return captions

    def get_caption_concat(
        self,
        separator: str = " ",
        mode: Literal["bfs", "dfs", "topological", "random"] = "bfs",
        remove_repeated_suffix: bool = False,
    ) -> str:
        captions = self.get_captions(
            with_bbox=False, mode=mode, remove_repeated_suffix=remove_repeated_suffix
        )
        return separator.join(captions)

    def get_graph_text_repr(
        self,
        root_id="image",
        caption_agg_mode: Literal["first", "concat"] = "first",
        concat_separator: str = " ",
        graph_traversal_mode: Literal["bfs", "dfs", "topological", "random"] = "bfs",
        remove_repeated_suffix: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        order = self.get_order(graph_traversal_mode, rng)
        vid_to_id = {vid: id for id, vid in enumerate(order)}
        vid_to_id[root_id] = vid_to_id[""]
        node_strings = []

        for id, vid in enumerate(order):
            vertex = self.vertex_dict[vid]

            # Get description of vertex
            description = ""
            for i, desc in enumerate(vertex.descs):
                text = desc.text
                if remove_repeated_suffix:
                    text = remove_repeated_suffix_func(text)
                if i == 0:
                    description = text
                    if caption_agg_mode == "first":
                        break
                else:
                    description += concat_separator + text

            # Get bbox string of vertex
            bbox = vertex.bbox.model_dump()
            bbox.pop("confidence", None)
            bbox_str = ""
            for k, v in bbox.items():
                bbox_str += f"{k}: {v:.7f}, "
            bbox_str = bbox_str.strip()[:-1]

            # Get complete node string
            node_string = ""
            node_string += f"Node #{id} {vid or root_id}\n"
            node_string += f"type: {vertex.label}\n"
            node_string += f"is_leaf: {vertex.is_leaf}\n"
            node_string += f"desc: {description}\n"
            node_string += "parents:"
            for edge in vertex.in_edges:
                pid = edge.source or root_id
                edge_text = edge.text
                node_string += f" #{vid_to_id[pid]}({pid}: {edge_text})"
            node_string += "\n"
            node_string += f"bbox: {bbox_str}\n"
            node_strings.append(node_string)

        return "\n".join(node_strings)

    """
    Subgraph
    """

    def _get_subgraph_vertices(
        self,
        root_id: Optional[str] = None,
        max_n_vertices: Optional[int] = None,
        max_depth: Optional[int] = None,
        mode: Literal["bfs", "dfs"] = "bfs",
        edge_shuffling: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> list[GbcVertexFull]:

        if rng is None:
            rng = np.random.default_rng()

        if root_id is None:
            to_process = self.roots
        else:
            vertex = self.vertex_dict[root_id]
            to_process = [vertex]
        n_subgraph_vertices = 0
        visited = set()
        vertices = []

        while len(to_process) != 0 and (
            max_n_vertices is None or n_subgraph_vertices < max_n_vertices
        ):
            if mode == "bfs":
                vertex = to_process.pop(0)
                edges = vertex.out_edges
            else:
                vertex = to_process.pop()
                edges = list(reversed(vertex.out_edges))
            if vertex.vertex_id in visited:
                continue
            visited.add(vertex.vertex_id)
            if max_depth is not None and self.node_depths[vertex.vertex_id] > max_depth:
                continue
            # It is important to use deepcopy to avoid undesired side effects
            vertices.append(deepcopy(vertex))
            if edge_shuffling:
                edges = rng.permutation(edges)
            for edge in edges:
                if edge.target not in visited:
                    to_process.append(self.vertex_dict[edge.target])
            n_subgraph_vertices += 1
        return vertices

    def _get_subgraph_vertices_from_root(
        self,
        max_n_vertices: Optional[int] = None,
        max_depth: Optional[int] = None,
        mode: Literal["bfs", "dfs", "random"] = "bfs",
        rng: Optional[np.random.Generator] = None,
    ) -> list[GbcVertexFull]:
        if max_n_vertices is None:
            max_n_vertices = len(self.vertices)
        order = self.get_order(mode, rng)
        vertices = []
        for vertex_id in order:
            if max_depth is not None and self.node_depths[vertex_id] > max_depth:
                continue
            if len(vertices) >= max_n_vertices:
                break
            vertices.append(deepcopy(self.vertex_dict[vertex_id]))
        return vertices

    def get_subgraph(
        self,
        max_n_vertices: Optional[int] = None,
        max_depth: Optional[int] = None,
        mode: Literal["bfs", "dfs", "random"] = "bfs",
        root_id: Optional[str] = None,
        edge_shuffling: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> "GbcGraphFull":
        if root_id is None and not edge_shuffling:
            vertices = self._get_subgraph_vertices_from_root(
                max_n_vertices, max_depth, mode
            )
        else:
            if mode == "random":
                raise ValueError("Random mode with general root not supported")
            vertices = self._get_subgraph_vertices(
                root_id, max_n_vertices, max_depth, mode, edge_shuffling, rng=rng
            )
        # For some reason deepcopy does not recompute cached properties
        new_graph_dict = self.model_dump()
        new_graph_dict["vertices"] = vertices
        new_graph = self.__class__.model_validate(new_graph_dict)
        return new_graph

    """
    Filtering out nodes/captions (in-place operations)
    """

    def drop_vertex(
        self,
        vertex_id: str,
        reset_cache: bool = True,
        keep_in_edges: bool = True,
        keep_out_edges: bool = True,
    ) -> Self:
        vertex_to_drop = self.vertex_dict[vertex_id]
        parents = [edge.source for edge in vertex_to_drop.in_edges]
        children = [edge.target for edge in vertex_to_drop.out_edges]

        new_vertices = []
        for vertex in self.vertices:
            if vertex.vertex_id == vertex_id:
                continue
            if vertex.vertex_id in parents:
                new_out_edges = []
                for edge in vertex.out_edges:
                    if edge.target != vertex_id:
                        new_out_edges.append(edge)
                # Make shortcut edge if dropped vertex is not a relation
                if vertex_to_drop.label != "relation" and keep_out_edges:
                    for edge in vertex_to_drop.out_edges:
                        # This is with labels of children of dropped vertex
                        edge_update = GbcEdge(
                            source=vertex.vertex_id, text=edge.text, target=edge.target
                        )
                        if edge_update not in new_out_edges:
                            new_out_edges.append(edge_update)
                vertex.out_edges = new_out_edges
            if vertex.vertex_id in children:
                new_in_edges = []
                for edge in vertex.in_edges:
                    if edge.source != vertex_id:
                        new_in_edges.append(edge)
                # Make shortcut edge if dropped vertex is not a relation
                if vertex_to_drop.label != "relation" and keep_in_edges:
                    for edge in vertex_to_drop.in_edges:
                        # This is with label of the dropped vertex
                        edge_update = GbcEdge(
                            source=edge.source, text=edge.text, target=vertex.vertex_id
                        )
                        if edge_update not in new_in_edges:
                            new_in_edges.append(edge_update)
                vertex.in_edges = new_in_edges
            new_vertices.append(vertex)
        self.vertices = new_vertices
        if reset_cache:
            self.reset_cache()
        return self

    def drop_vertices_by_type(
        self,
        vertex_types: list[str],
        keep_in_edges: bool = True,
        keep_out_edges: bool = True,
    ) -> Self:
        for vertex in self.vertices:
            if vertex.label in vertex_types:
                self.drop_vertex(
                    vertex.vertex_id,
                    reset_cache=False,
                    keep_in_edges=keep_in_edges,
                    keep_out_edges=keep_out_edges,
                )
        self.reset_cache()
        return self

    def drop_vertices_by_size(
        self,
        min_rel_width: float = None,
        min_rel_height: float = None,
        min_rel_size: float = None,
        max_rel_width: float = None,
        max_rel_height: float = None,
        max_rel_size: float = None,
        keep_in_edges: bool = True,
        keep_out_edges: bool = True,
    ) -> Self:
        add_kwargs = {
            "reset_cache": False,
            "keep_in_edges": keep_in_edges,
            "keep_out_edges": keep_out_edges,
        }
        for vertex in self.vertices:
            # Never drop image vertex
            if vertex.label == "image":
                continue
            bbox = vertex.bbox
            bbox_rel_width = bbox.right - bbox.left
            bbox_rel_height = bbox.bottom - bbox.top
            bbox_rel_size = bbox_rel_width * bbox_rel_height
            if min_rel_width is not None and bbox_rel_width < min_rel_width:
                self.drop_vertex(vertex.vertex_id, **add_kwargs)
            elif min_rel_height is not None and bbox_rel_height < min_rel_height:
                self.drop_vertex(vertex.vertex_id, **add_kwargs)
            elif min_rel_size is not None and bbox_rel_size < min_rel_size:
                self.drop_vertex(vertex.vertex_id, **add_kwargs)
            elif max_rel_width is not None and bbox_rel_width > max_rel_width:
                self.drop_vertex(vertex.vertex_id, **add_kwargs)
            elif max_rel_height is not None and bbox_rel_height > max_rel_height:
                self.drop_vertex(vertex.vertex_id, **add_kwargs)
            elif max_rel_size is not None and bbox_rel_size > max_rel_size:
                self.drop_vertex(vertex.vertex_id, **add_kwargs)
        self.reset_cache()
        return self

    def drop_composition_descendants(self) -> Self:
        def check_composition_children():
            for vertex in self.vertices:
                if vertex.label == "composition":
                    if vertex.out_edges:
                        return True
            return False

        while check_composition_children():
            for vertex in self.vertices:
                if vertex.label == "composition":
                    for edge in vertex.out_edges:
                        self.drop_vertex(edge.target, reset_cache=False)
            self.reset_cache()
        return self

    def drop_vertices_by_overlap_area(
        self,
        max_overlap_ratio: float = 0.5,
        keep_in_edges: bool = True,
        keep_out_edges: bool = True,
    ) -> Self:
        """
        Drop vertices such that vertices of the same depth overlap
        less than max_overlap_ratio
        """
        current_depth = 1
        while current_depth <= self.depth:
            nodes_current_depth = self.depth_to_nodes[current_depth]
            nodes_current_depth = sorted(
                nodes_current_depth,
                key=lambda vertex_id: self.vertex_dict[vertex_id].bbox.compute_area(),
            )
            dropped_vertices = set()
            for vertex_id in nodes_current_depth:
                if vertex_id in dropped_vertices:
                    continue
                vertex = self.vertex_dict[vertex_id]
                for other_vertex_id in nodes_current_depth:
                    if (
                        other_vertex_id == vertex_id
                        or other_vertex_id in dropped_vertices
                    ):
                        continue
                    other_vertex = self.vertex_dict[other_vertex_id]
                    if vertex.bbox.is_overlapped(
                        other_vertex.bbox,
                        threshold1=max_overlap_ratio,
                        threshold2=max_overlap_ratio,
                        or_overlap=True,
                    ):
                        self.drop_vertex(
                            other_vertex_id,
                            reset_cache=False,
                            keep_in_edges=keep_in_edges,
                            keep_out_edges=keep_out_edges,
                        )
                        dropped_vertices.add(other_vertex_id)
            if len(dropped_vertices) > 0:
                self.reset_cache()
            # Only go to next depth if no vertices were dropped
            else:
                current_depth += 1
        return self

    def drop_captions_by_type(
        self,
        caption_types: list[str],
        drop_empty_nodes: bool = True,
        keep_in_edges: bool = True,
        keep_out_edges: bool = True,
    ) -> Self:
        node_dropped = False
        for vertex in self.vertices:
            new_descs = []
            for desc in vertex.descs:
                if desc.label in caption_types or desc.full_label in caption_types:
                    continue
                new_descs.append(desc)
            vertex.descs = new_descs
            if drop_empty_nodes and len(vertex.descs) == 0:
                self.drop_vertex(
                    vertex.vertex_id,
                    reset_cache=False,
                    keep_in_edges=keep_in_edges,
                    keep_out_edges=keep_out_edges,
                )
                node_dropped = True
        if node_dropped:
            self.reset_cache()
        return self
