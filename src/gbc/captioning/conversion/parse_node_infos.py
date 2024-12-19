# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from copy import deepcopy

from gbc.utils import ImageCache
from gbc.data.bbox import union_bboxes
from ..primitives import EntityInfo, QueryResult, ActionInput, NodeInfo


def parse_node_info_list_per_image(
    node_infos: list[NodeInfo],
) -> tuple[dict[str, NodeInfo], dict[str, list[str]]]:

    # Map the first entity id to the node info
    id_to_node_info = dict()
    # For nodes with multiple entity infos, mapping the ids to the first one
    entity_id_mapping = dict()
    # Map potential composition nodes to their children
    composition_to_children = dict()
    # Store relational node infos for separate treatment
    relational_node_infos = []

    for node_info in node_infos:

        # Ignore query result with no description
        # These query results should still contain raw query response
        if len(node_info.query_result.descs) == 0:
            continue

        entity_infos = node_info.action_input.entity_info
        if not isinstance(entity_infos, list):
            entity_infos = [entity_infos]

        assert (
            len(entity_infos) > 0
        ), f"Image {node_info.img_path}: Empty entity info is not allowed"

        # Store relational node infos for separate treatment
        if entity_infos[0].label == "relation":
            relational_node_infos.append(node_info)
            continue

        # Map all ids to first entity id and map first entity id to node
        first_entity_id = entity_infos[0].entity_id
        id_to_node_info[first_entity_id] = node_info
        label = None
        for entity_info in entity_infos:
            assert entity_info.entity_id not in entity_id_mapping, (
                f"Image {node_info.img_path}: ",
                "Entity id should be unique, "
                f"but'{entity_info.entity_id}' is found in {entity_id_mapping}",
            )
            if label is None:
                label = entity_info.label
            assert label == entity_info.label, (
                f"Image {node_info.img_path}: ",
                "Collided entity infos should have the same label, ",
                f"but both {label} and {entity_info.label} are found",
            )
            entity_id_mapping[entity_info.entity_id] = [first_entity_id]
            # Update composition node information
            if "_" in entity_info.entity_id:
                left, right = entity_info.entity_id.rsplit("_", 1)
                if right.isdigit():
                    # Note that we always map to first entity id
                    if left in composition_to_children:
                        composition_to_children[left].append(first_entity_id)
                    else:
                        composition_to_children[left] = [first_entity_id]

    # Deal with numbered ids
    for composition_id in composition_to_children.keys():

        # This happens either because
        # - We have not performed composition query
        # - There is a single numbered bounding box left, so we do not
        # need composition node
        if composition_id not in entity_id_mapping:
            # This makes sure that whenever we point to a non-existing composition node,
            # we will point to its children entity node
            entity_id_mapping[composition_id] = composition_to_children[composition_id]

    # Deal with relation node infos
    for node_info in relational_node_infos:
        new_nodes = _split_relation_node(node_info, id_to_node_info, entity_id_mapping)
        # This add relation nodes and updates parent node of these nodes
        id_to_node_info.update(new_nodes)
        entity_id_mapping.update(
            {id: [id] for id in new_nodes.keys() if id not in entity_id_mapping}
        )

    return id_to_node_info, entity_id_mapping


def _split_relation_node(
    node_info: NodeInfo,
    id_to_node_info: dict[str, NodeInfo],
    entity_id_mapping: dict[str, list[str]],
) -> list[NodeInfo]:

    def get_node(entity_id: str) -> NodeInfo:
        node_id = entity_id_mapping[entity_id][0]
        node = deepcopy(id_to_node_info[node_id])
        return node, node_id

    entity_info = node_info.action_input.entity_info
    if isinstance(entity_info, list):
        raise ValueError("We do not expect node collision for relation query")

    # This is either image or entity node
    relation_parent_node, parent_node_id = get_node(entity_info.entity_id)

    new_nodes = dict()

    for desc, associated_ids in node_info.query_result.descs:
        associated_ids = sorted(associated_ids)
        new_entity_id = "[" + "|".join(associated_ids) + "]"

        # Create a node for each set of children
        if new_entity_id not in new_nodes:

            children_entities = []
            children_bboxes = []
            children_texts = []

            for id in associated_ids:
                child, _ = get_node(id)
                # This works by design
                child_text = id.rsplit("_", 1)[-1]

                child_entity_info = EntityInfo(
                    label=child.label,  # This is not used
                    entity_id=id,
                    text=child_text,
                )
                # The second part is RefPosition
                # We don't need it here as this is not used later
                children_entities.append((child_entity_info, []))
                # The reason why we need child
                children_bboxes.append(child.action_input.bbox)
                children_texts.append(child_text)

            new_entity_info = EntityInfo(
                label="relation",
                entity_id=new_entity_id,
                text=None,  # This is not used
            )
            new_action_input = ActionInput(
                image=ImageCache(img_path=node_info.img_path),
                entity_info=new_entity_info,
                bbox=union_bboxes(children_bboxes),
            )
            query_result = QueryResult(
                descs=[], entities=children_entities, raw=node_info.query_result.raw
            )
            new_node = NodeInfo(
                action_input=new_action_input, query_result=query_result
            )
            new_nodes[new_entity_id] = new_node

            # Update out edges of parent node
            # (multiple edges with different texts point to new relation node)
            for child_text in children_texts:
                new_entity_info = EntityInfo(
                    label="relation", entity_id=new_entity_id, text=child_text
                )
                relation_parent_node.query_result.entities.append((new_entity_info, []))
            new_nodes[parent_node_id] = relation_parent_node

        new_node = new_nodes[new_entity_id]
        new_node.query_result.descs.append((desc, [new_entity_id]))

    return new_nodes
