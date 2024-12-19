# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import re
from typing import TypeVar, Optional, Literal, Union
from pydantic import BaseModel
from tqdm import tqdm

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

from gbc.utils import get_gbc_logger
from gbc.data.caption import Description, Caption, get_bag_of_words_captions
from gbc.data.graph import GbcGraph, GbcVertexFull, GbcGraphFull


DataType = TypeVar(
    "DataType", Description, GbcVertexFull, GbcGraphFull, list[GbcGraphFull]
)


def gbc_clip_filter(
    data: DataType,
    # Parameters for filtering of individual captions
    max_n_tokens: Optional[int] = None,
    split_rather_than_filter: bool = True,
    remove_truncation: bool = True,
    exclude_labels: list[str] = ["hardcode"],
    # For random filtering
    random_filtering_probs: dict[str, float] = dict(),
    seed: int = 486,
    # Clip filtering
    clip_names: list[str] = [
        "dfn5b-h-patch14-378",
        "openai-l-patch14-336",
    ],
    min_clip_scores: dict[str, float] = {
        "short-global": 0.2,
        "detail-entity": 0.2,
    },
    # Vertex level
    add_bag_of_words: bool = True,
    remove_edge_with_no_corr: bool = True,
    # Image filtering
    filter_image_using_short_clip_scores: bool = True,
    must_include_labels: list[str] = [],
    filter_after_selection: bool = False,
    max_n_vertices_per_graph: Optional[int] = None,
    node_selection_mode: Literal["bfs", "dfs", "random"] = "bfs",
    # Parameters for filtering of the entire dataset
    max_n_vertices: Optional[int] = None,
) -> Optional[DataType]:
    # Instantiate the GbcClipFilter class
    gbc_filter = GbcClipFilter(
        max_n_tokens=max_n_tokens,
        split_rather_than_filter=split_rather_than_filter,
        remove_truncation=remove_truncation,
        exclude_labels=exclude_labels,
        random_filtering_probs=random_filtering_probs,
        seed=seed,
        clip_names=clip_names,
        min_clip_scores=min_clip_scores,
        add_bag_of_words=add_bag_of_words,
        remove_edge_with_no_corr=remove_edge_with_no_corr,
        filter_image_using_short_clip_scores=filter_image_using_short_clip_scores,
        must_include_labels=must_include_labels,
        filter_after_selection=filter_after_selection,
        max_n_vertices_per_graph=max_n_vertices_per_graph,
        node_selection_mode=node_selection_mode,
        max_n_vertices=max_n_vertices,
    )
    # Call the filter on the provided data
    return gbc_filter(data)


class FilteringStatistics(BaseModel):

    n_included_graphs: int = 0
    n_included_vertices: int = 0
    n_included_captions: int = 0
    n_included_edges: int = 0

    # Excluded (may be due to vertex limit constraint)
    n_excluded_vertices: int = 0
    n_excluded_edges: int = 0

    # Filtered
    n_filtered_graphs: int = 0
    n_filtered_vertices: int = 0
    n_filtered_captions: int = 0

    # Others
    n_split_captions: int = 0
    n_added_captions_from_splits: int = 0
    # This can be smaller than `n_included_captions` because some
    # captions are split into multiple captions
    # This can be larger than `n_included_captions` because we may
    # select just subgraph
    n_not_filtered_captions: int = 0
    n_bagofwords_captions: int = 0

    max_n_vertices_reached: bool = False

    def update(self, other: "FilteringStatistics"):
        for attr in other.__fields__.keys():
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))

    def update_with_graph_difference(
        self, original_graph: GbcGraphFull, filtered_graph: Optional[GbcGraphFull]
    ) -> None:
        if filtered_graph is None:
            self.n_filtered_graphs += 1
            self.n_excluded_vertices += original_graph.n_vertices
            self.n_excluded_edges += original_graph.n_edges
        else:
            self.n_included_graphs += 1
            self.n_included_vertices += filtered_graph.n_vertices
            self.n_included_captions += filtered_graph.n_captions
            self.n_included_edges += filtered_graph.n_edges
            self.n_excluded_vertices += (
                original_graph.n_vertices - filtered_graph.n_vertices
            )
            self.n_excluded_edges += original_graph.n_edges - filtered_graph.n_edges


def string_in_desc_texts(string: str, desc_texts: list[str]) -> bool:
    """
    Checks if the given string appears in any of the description texts.,
    considering word boundaries to ensure exact matches.

    Parameters
    ----------
    string
        The string to search for.
    desc_texts
        A list of description texts to search in.

    Returns
    -------
    bool
        True if the entity_text is found in any of the ``desc_texts``, False otherwise.
    """
    # Create a pattern that matches the entity text only
    # if it's surrounded by word boundaries
    pattern = r"\b" + re.escape(string) + r"\b"

    for desc_text in desc_texts:
        # Find all occurrences of the entity text in the description text,
        # considering word boundaries
        if re.search(pattern, desc_text, re.IGNORECASE):
            return True  # Return True if a match is found
    return False  # Return False if no matches are found


class GbcClipFilter(BaseModel):
    """
    A class for filtering captions, vertices, and graphs

    -----------------------------------------------------------------------------
    Arguments for Basic Caption Filtering:
    -----------------------------------------------------------------------------

    max_n_tokens (Optional[int]):
        The maximum number of tokens allowed per caption.
    split_rather_than_filter: bool
        If True, we try to split too long captions into shorter ones that
        comply with the maximum number of tokens.
        Defaults to True.
    remove_truncation: bool
        If True, we filter out captions with truncation.
        Defaults to True.
    exclude_labels: list[str]
        We filter out captions whose labels or full labels are in this list.
        Examples include "detail", "short-entity", "detail-global", "relation", etc.

    -----------------------------------------------------------------------------
    Arguments for iid Random Caption Filtering:
    -----------------------------------------------------------------------------

    random_filtering_probs: dict[str, float]
        We randomly filter out a part of captions whose labels or full labels
        are in this list. The values indicate the probability of being filtered.
    seed: int
        Random seed.

    -----------------------------------------------------------------------------
    Arguments for Clip-Based Caption Filtering:
    -----------------------------------------------------------------------------

    clip_names: list[str]
        The clip scores used for filtering is computed as average of clip scores
        from these models.
    min_clip_scores: dict[str, float]
        We filter out captions whose clip scores are below the threshold.
        The key should be either label or full label of caption.

    -----------------------------------------------------------------------------
    Arguments for Post-Processing a Vertex after Caption Dropping:
    -----------------------------------------------------------------------------

    add_bag_of_words: bool
        Whether to add bag-of-word captions if some edges are not mapped to
        any caption after filtering. Defaults to True.
    remove_edge_with_no_corr: bool
        Whether to remove edges whose edge labels are not mapped to any caption
        after filtering. Defaults to True.

    -----------------------------------------------------------------------------
    Arguments for Image Filtering:
    -----------------------------------------------------------------------------

    filter_image_using_short_clip_scores: bool
        If true, we always filter image vertex whose short captions have too low
        clip scores.
    must_include_labels: list[str]
        The caption label that must be included at the image node.
        If all captions of this label are filtered out, the image is filtered out.
    filter_after_selection (bool, optional):
        If True, apply filtering after node selection. Defaults to False.
    max_n_vertices_per_graph (Optional[int]):
        The maximum number of vertices allowed per graph after processing.
    node_selection_mode (Literal["bfs", "dfs", "random"], optional):
        The node selection strategy for subgraph extraction. Defaults to "bfs".

    -----------------------------------------------------------------------------
    Arguments for Dataset Filtering:
    -----------------------------------------------------------------------------

    max_n_vertices (Optional[int]):
        The maximum number of vertices allowed across all returned graphs.
        If the total exceeds this, filtering stops.
    """

    # Parameters for filtering of individual captions
    max_n_tokens: Optional[int] = None
    split_rather_than_filter: bool = True
    remove_truncation: bool = True
    exclude_labels: list[str] = ["hardcode"]

    # For random filtering
    random_filtering_probs: dict[str, float] = dict()
    seed: int = 486

    # Clip filtering
    clip_names: list[str] = [
        "dfn5b-fm-b-patch32-224",
        "dfn5b-h-patch14-378",
        "openai-l-patch14-336",
    ]
    min_clip_scores: dict[str, float] = {
        "short-global": 0.2,
        "detail-entity": 0.2,
    }

    # Vertex level
    add_bag_of_words: bool = True
    remove_edge_with_no_corr: bool = True

    # Image filtering
    filter_image_using_short_clip_scores: bool = True
    must_include_labels: list[str] = []
    filter_after_selection: bool = False
    max_n_vertices_per_graph: Optional[int] = None
    node_selection_mode: Literal["bfs", "dfs", "random"] = "bfs"

    # Parameters for filtering of the entire dataset
    max_n_vertices: Optional[int] = None

    def model_post_init(self, context):
        nltk.download("punkt_tab", quiet=True)
        self._rng = np.random.default_rng(self.seed)
        self._logger = get_gbc_logger()

    def __call__(self, data: DataType, **kwargs) -> Optional[DataType]:
        return self.filter(data, **kwargs)

    def filter(
        self, data: DataType, **kwargs
    ) -> Union[Optional[DataType], Optional[list[DataType]]]:
        if (
            isinstance(data, list)
            and (len(data) > 0)
            and isinstance(data[0], GbcGraphFull)
        ):
            return self.filter_graphs(data, **kwargs)
        elif isinstance(data, GbcGraphFull):
            return self.filter_graph(data)
        elif isinstance(data, GbcVertexFull):
            return self.filter_vertex(data)
        elif isinstance(data, Caption):
            return self.filter_caption(data)
        else:
            raise ValueError(f"Unsupported type {type(data)}")

    def filter_graphs(
        self,
        gbc_graphs: list[GbcGraphFull],
        return_statistics: bool = False,
        verbose: bool = False,
        filtering_statistics: Optional[FilteringStatistics] = None,
    ) -> Union[list[GbcGraphFull], tuple[list[GbcGraphFull], int]]:
        """
        Filters and processes a list of GBC graphs according to specified
        criteria defined in this class.

        Parameters
        ----------
        gbc_graphs
            A list of GBC graphs to be processed.
        return_statistics
            If True, returns also the filtering statistics. Defaults to False.
        verbose
            If True, enables progress messages. Defaults to False.
        filtering_statistics
            An optional filtering statistics object to update.

        Returns
        -------
        Union[list[GbcGraphFull], tuple[list[GbcGraphFull], FilteringStatistics]]:
            Depending on the `return_statistics` flag,
            either a list of processed GBC graphs
            or a tuple of this list and filtering statistics.
        """
        if filtering_statistics is None:
            filtering_statistics = FilteringStatistics()
        gbc_graphs_filtered = []
        for gbc_graph in tqdm(
            gbc_graphs, desc="Filtering and extracting subgraph", disable=not verbose
        ):
            filtered_graph, filter_statistics_graph = self.filter_graph(gbc_graph)
            if filtered_graph is not None:
                if (
                    self.max_n_vertices is not None
                    and filtering_statistics.n_included_vertices
                    + filter_statistics_graph.n_included_vertices
                    > self.max_n_vertices
                ):
                    filtering_statistics.max_n_vertices_reached = True
                    break
                gbc_graphs_filtered.append(filtered_graph)
            filtering_statistics.update(filter_statistics_graph)

        if return_statistics:
            return gbc_graphs_filtered, filtering_statistics
        return gbc_graphs_filtered

    def filter_caption(
        self, caption: Caption
    ) -> tuple[Optional[list[Caption]], FilteringStatistics]:
        """
        Filtering of a single caption based on
        - caption type
        - caption length
        - clip score
        """
        filtering_statistics = FilteringStatistics()

        results = [caption]

        to_filter = False

        # Filter empty captions
        if not to_filter and caption.text.strip() == "":
            self._logger.warning("Empty caption detected")
            to_filter = True

        # Filter based on caption type
        if not to_filter and self.exclude_labels is not None:
            for label in self.exclude_labels:
                if (label == caption.label) or (label == caption.full_label):
                    to_filter = True
                    break

        # Random filtering
        if not to_filter and self.random_filtering_probs:
            if caption.full_label in self.random_filtering_probs:
                if self._rng.random() < self.random_filtering_probs[caption.full_label]:
                    to_filter = True
            elif caption.label in self.random_filtering_probs:
                if self._rng.random() < self.random_filtering_probs[caption.label]:
                    to_filter = True

        # Filter captions with too long sentences
        if not to_filter and self.remove_truncation and caption.clip_scores.truncation:
            to_filter = True

        # Max token filtering / splitting
        if not to_filter and self.max_n_tokens is not None:
            n_tokens = caption.get_statistics().n_tokens
            if n_tokens > self.max_n_tokens:
                if self.split_rather_than_filter:
                    caption_split = sent_tokenize(caption.text)
                    # We discard single sentence with more than X tokens
                    results, _ = get_bag_of_words_captions(
                        caption_split,
                        type(caption),
                        max_n_tokens=self.max_n_tokens,
                        splitter=" ",
                    )
                    for desc_new in results:
                        desc_new.label = caption.label
                        if hasattr(desc_new, "full_label"):
                            desc_new.full_label = caption.full_label
                        if hasattr(desc_new, "clip_scores"):
                            desc_new.clip_scores = caption.clip_scores
                        if hasattr(desc_new, "toxicity_scores"):
                            desc_new.toxicity_scores = caption.toxicity_scores
                    filtering_statistics.n_split_captions = 1
                    filtering_statistics.n_added_captions_from_splits = len(results) - 1
                else:
                    to_filter = True

        # Clip filtering
        if not to_filter and (
            caption.full_label in self.min_clip_scores
            or caption.label in self.min_clip_scores
        ):
            clip_score = self.get_clip_score(caption.clip_scores.scores)
            if (
                caption.full_label in self.min_clip_scores
                and clip_score < self.min_clip_scores[caption.full_label]
            ):
                to_filter = True
            elif (
                caption.label in self.min_clip_scores
                and clip_score < self.min_clip_scores[caption.label]
            ):
                to_filter = True

        if to_filter:
            filtering_statistics.n_split_captions = 0
            filtering_statistics.n_added_captions_from_splits = 0
            filtering_statistics.n_filtered_captions = 1
            return None, filtering_statistics

        filtering_statistics.n_not_filtered_captions = 1
        return results, filtering_statistics

    def get_clip_score(self, scores: Optional[dict[str, float]]) -> float:
        if scores is None:
            # will never be filtered
            return float("inf")
        values = [
            score for clip_name, score in scores.items() if clip_name in self.clip_names
        ]
        if len(values) > 0:
            return np.mean(values)
        return float("inf")

    def filter_vertex(
        self, vertex: GbcVertexFull, filtered: dict[str, Optional[GbcVertexFull]]
    ) -> tuple[Optional[GbcVertexFull], FilteringStatistics]:
        """
        Note that this function creates new vertex and should not modify the original
        vertex. ``filtered`` is the dictionary of processed vertices and can be updated
        in case where ``in_edges`` are changed
        """
        filtering_statistics = FilteringStatistics()

        out_edge_update = []
        remaining_texts = set()
        for edge in vertex.out_edges:
            assert (
                edge.target in filtered
            ), "all the descendants must be processed first during filtering"
            if filtered[edge.target] is not None:
                out_edge_update.append(edge)
                remaining_texts.add(edge.text)

        descs_update = []
        descs_update_texts = []
        for desc in vertex.descs:
            # Filter out short global captions if the score of short caption is too low
            if (
                self.filter_image_using_short_clip_scores
                and desc.label == "short"
                and vertex.label == "image"
                and "short-image" in self.min_clip_scores
                and self.get_clip_score(desc.clip_scores.scores)
                < self.min_clip_scores["short-image"]
            ):
                filtering_statistics.n_filtered_vertices += 1
                return None, filtering_statistics
            descs_filtered, filtering_statistics_caption = self.filter_caption(desc)
            filtering_statistics.update(filtering_statistics_caption)
            if descs_filtered is not None:
                descs_update.extend(descs_filtered)
                descs_update_texts.extend([desc.text for desc in descs_filtered])

        # Filter out image nodes if they have neither short,
        # detail nor original captions
        if vertex.label == "image":
            must_include = set(self.must_include_labels)
            should_filter = True
            for desc in descs_update:
                must_include.discard(desc.label)
                if desc.label in ["short", "detail", "original"]:
                    should_filter = False
            if should_filter or len(must_include) > 0:
                filtering_statistics.n_filtered_vertices += 1
                return None, filtering_statistics

        # Check if there are edge texts that cannot be mapped to captions
        remaining_texts_update = []
        for edge_text in remaining_texts:
            if string_in_desc_texts(edge_text, descs_update_texts):
                continue
            remaining_texts_update.append(edge_text)
        remaining_texts = remaining_texts_update

        # Create bag of word caption if some texts for outgoing edges
        # do not appear in the captions anymore after filtering
        if len(remaining_texts) > 0 and self.add_bag_of_words:
            bag_of_words_captions, should_remove_texts = get_bag_of_words_captions(
                remaining_texts,
                type(vertex.descs[0]),
                max_n_tokens=self.max_n_tokens,
                vertex_label=vertex.label,
            )
            for desc in bag_of_words_captions:
                descs_update.append(desc)
            filtering_statistics.n_bagofwords_captions += len(bag_of_words_captions)
        else:
            should_remove_texts = set(remaining_texts)

        # If we create bag of word, this only happens if one edge text
        # contains for example more than 77 tokens
        # The in edges of the children node also need to be updated
        if len(should_remove_texts) > 0 and self.remove_edge_with_no_corr:
            out_edge_update_new = []
            for edge in out_edge_update:
                if edge.text in should_remove_texts:
                    child_vertex = filtered[edge.target]
                    in_edge_update = []
                    for in_edge in child_vertex.in_edges:
                        if (
                            in_edge.source != vertex.vertex_id
                            or in_edge.text != edge.text
                        ):
                            in_edge_update.append(in_edge)
                    child_vertex.in_edges = in_edge_update
                else:
                    out_edge_update_new.append(edge)
            out_edge_update = out_edge_update_new

        if len(descs_update) == 0:
            filtering_statistics.n_filtered_vertices += 1
            return None, filtering_statistics

        vertex_class = type(vertex)
        vertex_dict = vertex.model_dump()
        vertex_dict["descs"] = descs_update
        vertex_dict["out_edges"] = out_edge_update

        return vertex_class.model_validate(vertex_dict), filtering_statistics

    def get_filtered_vertices_aux(
        self,
        graph: GbcGraphFull,
        root: Optional[GbcVertexFull] = None,
        filtered: Optional[dict[str, Optional[GbcVertexFull]]] = None,
    ) -> dict[str, Optional[GbcVertexFull]]:
        filtering_statistics = FilteringStatistics()

        if filtered is None:
            filtered = dict()
        filtered_to_update = dict()
        if root is None:
            for root in graph.roots:
                new_filtered, new_filtering_statistics = self.get_filtered_vertices_aux(
                    graph, root, filtered | filtered_to_update
                )
                # This is to avoid in-place modification
                filtered_to_update.update(new_filtered)
                filtering_statistics.update(new_filtering_statistics)
        else:
            # If root already in filtered there is nothing to do
            if root.vertex_id not in filtered:
                for edge in root.out_edges:
                    vertex_target = graph.vertex_dict[edge.target]
                    new_filtered, new_filtering_statistics = (
                        self.get_filtered_vertices_aux(
                            graph, vertex_target, filtered | filtered_to_update
                        )
                    )
                    filtered_to_update.update(new_filtered)
                    filtering_statistics.update(new_filtering_statistics)
                # The actual place where filtering happens
                root_filtered, filtering_statistics_vertex = self.filter_vertex(
                    root, filtered | filtered_to_update
                )
                filtered_to_update[root.vertex_id] = root_filtered
                filtering_statistics.update(filtering_statistics_vertex)
        return filtered_to_update, filtering_statistics

    def get_filtered_vertices(self, graph: GbcGraphFull) -> dict[str, GbcVertexFull]:
        filterd_vertices, filtering_statistics = self.get_filtered_vertices_aux(graph)
        # Note that by design of vertex filtering we do not need to update
        # the edges anymore
        remained_vertices = {
            vertex_id: vertex
            for vertex_id, vertex in filterd_vertices.items()
            if vertex is not None
        }
        return remained_vertices, filtering_statistics

    def filter_graph(self, graph: GbcGraph) -> Optional[GbcGraphFull]:

        if not isinstance(graph, GbcGraphFull):
            graph = GbcGraphFull.from_gbc_graph(graph)

        if self.filter_after_selection and self.max_n_vertices_per_graph is not None:
            graph = graph.get_subgraph(
                max_n_vertices=self.max_n_vertices_per_graph,
                mode=self.node_selection_mode,
                rng=self._rng,
            )

        filtered_vertices, filtering_statistics = self.get_filtered_vertices(graph)

        # Root node is special and may be filtered even if there were
        # children, in which case we should filter the entire graph
        if graph.roots[0].vertex_id not in filtered_vertices:
            # We initialize from empty filtering statistics in case
            # the graph is filtered, to get the right number of
            # `n_added_captions_from_splits` and `n_bagofwords_captions`
            filtering_statistics = FilteringStatistics()
            filtering_statistics.update_with_graph_difference(
                original_graph=graph, filtered_graph=None
            )
            return None, filtering_statistics

        graph_dict = graph.model_dump()

        graph_dict["vertices"] = filtered_vertices.values()
        filtered_graph = GbcGraphFull.model_validate(graph_dict)

        if (
            not self.filter_after_selection
            and self.max_n_vertices_per_graph is not None
        ):
            filtered_graph = filtered_graph.get_subgraph(
                max_n_vertices=self.max_n_vertices_per_graph,
                mode=self.node_selection_mode,
                rng=self._rng,
            )
        filtering_statistics.update_with_graph_difference(
            original_graph=graph, filtered_graph=filtered_graph
        )
        if self.max_n_vertices_per_graph is None and not (
            filtering_statistics.n_included_captions
            == filtering_statistics.n_not_filtered_captions
            + filtering_statistics.n_bagofwords_captions
            + filtering_statistics.n_added_captions_from_splits
        ):
            print(filtering_statistics.dict())
            raise ValueError("Incorrect filtering statistics")
        return filtered_graph, filtering_statistics
