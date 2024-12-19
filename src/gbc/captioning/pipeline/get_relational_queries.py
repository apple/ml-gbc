# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from typing import Optional
from collections import defaultdict

import numpy as np

from gbc.utils import ImageCache
from gbc.texts import plural_to_singular
from gbc.data.bbox import union_bboxes
from gbc.data.graph.gbc_graph_full import GbcVertexFull, GbcGraphFull

from ..primitives import (
    EntityInfo,
    ActionInputWithEntities,
    ActionInputWithBboxes,
    ActionInputPair,
)
from ..auto_actions import AutoCompositionQuery, AutoRelationQuery


class RelationalQueryManager(object):

    def __init__(self, graph: GbcGraphFull):
        self.vertex_dict = graph.vertex_dict
        self.img_path = graph.img_path
        assert self.img_path is not None

    def _get_relational_queries_for_vertex(
        self,
        vertex: GbcVertexFull,
        rng: np.random.Generator,
    ) -> tuple[list[ActionInputPair], list[ActionInputPair]]:

        has_relation_children = False
        text_to_ids_dict = defaultdict(list)

        # First map out texts to their ids
        for out_edge in vertex.out_edges:
            assert (
                out_edge.target in self.vertex_dict
            ), f"Vertex {out_edge.target} not found in graph"
            target_vertex = self.vertex_dict[out_edge.target]
            if target_vertex.label == "image":
                raise ValueError("Image node should not be a child of another node")
            # Relation node will not be children of relation or composition node
            elif target_vertex.label == "relation":
                has_relation_children = True
            # The case of composition or entity node
            else:
                text_to_ids_dict[out_edge.text].append(out_edge.target)

        # We next map the ids to their corresponding texts
        ids_to_texts_dict = defaultdict(list)

        for text, ids in text_to_ids_dict.items():
            ids = tuple(sorted(ids))
            ids_to_texts_dict[ids].append(text)

        # Each set of ids corresponds either to a child entity node or
        # a(n) (imaginary) child composition node
        to_relate_entities = []
        compositional_contents = []

        for ids, texts in ids_to_texts_dict.items():
            assert len(texts) > 0 and len(ids) > 0
            # Randomly choose one text when multiple texts are mapped
            # to the same set of nodes
            text = rng.choice(texts)
            entity_id = f"{vertex.vertex_id}_{text}" if vertex.vertex_id else text
            if len(ids) == 1:
                target_vertex = self.vertex_dict[ids[0]]
                assert (
                    target_vertex.label in ["entity", "composition"]
                    and target_vertex.vertex_id == ids[0]
                )
                # Use entity_id instead of ids[0] so that we can get the text later
                entity_info = EntityInfo(
                    entity_id=entity_id,
                    label=target_vertex.label,
                    text=text,
                )
                to_relate_entities.append(entity_info)
            else:
                entity_id = f"{vertex.vertex_id}_{text}" if vertex.vertex_id else text
                entity_info = EntityInfo(
                    entity_id=entity_id,
                    label="composition",
                    text=text,
                )
                to_relate_entities.append(entity_info)

                # Preparing for composition queries

                # The case of composition node collision
                # The two composition nodes have the same parent and children
                if len(texts) > 1:
                    entity_info = [entity_info]
                    for text_alt in texts:
                        if text_alt == text:
                            continue
                        entity_id = (
                            f"{vertex.vertex_id}_{text_alt}"
                            if vertex.vertex_id
                            else text_alt
                        )
                        entity_info.append(
                            EntityInfo(
                                entity_id=entity_id,
                                label="composition",
                                text=text_alt,
                            )
                        )

                # Children of composition node
                component_entity_infos = []
                component_bboxes = []
                for id in ids:
                    target_vertex = self.vertex_dict[id]
                    assert (
                        target_vertex.label == "entity"
                        and target_vertex.vertex_id == id
                    )
                    component_entity = EntityInfo(
                        entity_id=target_vertex.vertex_id,
                        label=target_vertex.label,
                        text=text,
                    )
                    component_entity_infos.append(component_entity)
                    component_bboxes.append(target_vertex.bbox)

                # Sort by entity_id
                sorted_idxs = np.argsort([e.entity_id for e in component_entity_infos])
                component_entity_infos_sorted = []
                component_bboxes_sorted = []
                for new_idx, old_idx in enumerate(sorted_idxs, start=1):
                    component_entity = component_entity_infos[old_idx]
                    # This is to match the text with how they are
                    # represented in composition captions
                    component_entity.text = (
                        plural_to_singular(component_entity.text) + f" {new_idx}"
                    )
                    component_entity_infos_sorted.append(
                        component_entity_infos[old_idx]
                    )
                    component_bboxes_sorted.append(component_bboxes[old_idx])

                compositional_contents.append(
                    (
                        entity_info,
                        component_entity_infos_sorted,
                        component_bboxes_sorted,
                    )
                )

        image = ImageCache(img_path=self.img_path)

        # Set relation query
        if not has_relation_children and len(to_relate_entities) > 1:
            entity_info = EntityInfo(
                entity_id=vertex.vertex_id,
                label="relation",
                text=None,  # This is not used
            )
            action_input_relation = ActionInputWithEntities(
                image=image,
                entity_info=entity_info,
                bbox=vertex.bbox,
                entities=to_relate_entities,
            )
            relation_query = ActionInputPair(
                action_class=AutoRelationQuery, action_input=action_input_relation
            )
            relation_queries = [relation_query]
        else:
            relation_queries = []

        # Set composition queries
        composition_queries = []
        for entity_info, entities, bboxes in compositional_contents:
            action_input_composition = ActionInputWithBboxes(
                image=image,
                entity_info=entity_info,
                bbox=union_bboxes(bboxes),
                entities=entities,
                bboxes=bboxes,
            )
            composition_queries.append(
                ActionInputPair(
                    action_class=AutoCompositionQuery,
                    action_input=action_input_composition,
                )
            )

        return composition_queries, relation_queries

    def get_relational_queries(
        self, rng: Optional[np.random.Generator] = None
    ) -> tuple[list[ActionInputPair], list[ActionInputPair]]:
        if rng is None:
            rng = np.random.default_rng()
        composition_queries = []
        relation_queries = []
        for vertex in self.vertex_dict.values():
            if vertex.label in ["composition", "relation"]:
                continue
            composition_queries_for_vertex, relation_queries_for_vertex = (
                self._get_relational_queries_for_vertex(vertex, rng)
            )
            composition_queries.extend(composition_queries_for_vertex)
            relation_queries.extend(relation_queries_for_vertex)
        return composition_queries, relation_queries


def get_relational_queries(
    graph: GbcGraphFull, rng: Optional[np.random.Generator] = None
) -> tuple[list[ActionInputPair], list[ActionInputPair]]:
    """
    Infer relation and composition queries to complete from the given GBC graph.

    .. warning::

       The artifacts (i.e. ``node_infos``) obtain from relation and composition
       queries themselves are not sufficient for constructing a GBC graph.
       They need to be used with the original ``node_infos``; see
       :meth:`GbcPipeline.run_relational_captioning() \
               <gbc.captioning.pipeline.pipeline.GbcPipeline.run_relational_captioning>`
       for an example of how to do so.
       In the future, we might consider implementing parsing from GBC graphs
       and additional node infos to avoid the need for original node infos.

    Parameters
    ----------
    graph
        The GBC graph.
    rng
        The random generator used to select the text when multiple out-edge texts
        from a node are mapped to the same set of children.
        A default random generator is used if not provided.

    Returns
    -------
    relation_queries
        The relation queries to be performed to complete the GBC graph.
    composition_queries
        The composition queries to be performed to complete the GBC graph.
    """
    if rng is None:
        rng = np.random.default_rng()
    relational_query_manager = RelationalQueryManager(graph)
    return relational_query_manager.get_relational_queries(rng)
