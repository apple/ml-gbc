# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import math
from pydantic import BaseModel
from typing import Optional
from random import choice

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree

from gbc.data import Bbox
from ..primitives.io_unit import EntityInfo


class MSTNode(BaseModel):
    """
    A node in the minimum spanning tree.

    This class implements a Euclidean minimum spanning tree for bounding
    boxes that are involved in composition queries.
    By selecting a random node as the root and traversing the tree with DFS,
    it generates hints based on geometric relations between the bounding boxes.

    Attributes
    ----------
    entity_info: EntityInfo
        The entity information of the node.
        We retrieve its text to craft the descriptions.
    bbox: Bbox
        The bounding box of the node, relative to the entire image.
        It is used to construct the minimum spanning tree.
    rel_bbox: Bbox
        The bounding box of the node, relative to the cropped image.
        It is used to determine whether a node is at extremity or not.
    children: list[MSTNode]
        The children of the node.
    """

    entity_info: EntityInfo
    bbox: Bbox
    rel_bbox: Bbox
    children: list["MSTNode"] = []

    def add_child(self, child: "MSTNode"):
        self.children.append(child)

    def get_all_bboxes(self, btype="rel"):
        if btype == "rel":
            bboxes = [self.rel_bbox]
        else:
            bboxes = [self.bbox]
        for child in self.children:
            bboxes.extend(child.get_all_bboxes())
        return bboxes

    def get_tree_descriptions(
        self, all_bboxes: Optional[list[Bbox]] = None
    ) -> tuple[list[str], list[str]]:
        """
        Get the description of the minimum spanning tree and the texts at each node.

        Parameters
        ----------
        all_bboxes
            The list of all relative bounding boxes in the tree.

        Returns
        -------
        descriptions
            The list of hard-coded hints to describe the relation
            between the objects in the tree.
        node_texts
            The list of texts at each node in the tree.
        """
        descriptions = []
        node_texts = [self.entity_info.text]
        if all_bboxes is None:
            all_bboxes = self.get_all_bboxes()

        # Get the description of the current node, if applicable
        node_description = self.get_node_description(all_bboxes)
        if node_description:
            descriptions.insert(
                0, node_description
            )  # Insert at the beginning(node_description)

        # Perform DFS to traverse child nodes and get their descriptions
        for child in self.children:
            # Get the edge description for the child node
            edge_description = self.get_edge_description(child)
            if edge_description:
                descriptions.append(edge_description)

            # Recursively get descriptions from the child node's subtree
            child_descriptions, child_node_texts = child.get_tree_descriptions(
                all_bboxes
            )
            descriptions.extend(child_descriptions)
            node_texts.extend(child_node_texts)

        return descriptions, node_texts

    def get_node_description(self, bboxes, margin_ths=0.1):
        bbox = self.rel_bbox
        center_x = (bbox.left + bbox.right) / 2
        center_y = (bbox.top + bbox.bottom) / 2

        # Define threshold for center region
        center_ths_low, center_ths_high = 0.4, 0.6

        # Determine the horizontal and vertical positions
        horizontal, vertical = None, None

        # Find extremities among all bboxes
        all_centers_x = [(bbox_.left + bbox_.right) / 2 for bbox_ in bboxes]
        all_centers_y = [(bbox_.top + bbox_.bottom) / 2 for bbox_ in bboxes]

        # Horizontal positioning logic
        if center_x == min(all_centers_x) and bbox.right < 0.5:
            horizontal = "leftmost"
        elif center_x == max(all_centers_x) and bbox.left > 0.5:
            horizontal = "rightmost"
        elif bbox.left < margin_ths and bbox.right < 0.5:
            horizontal = "left"
        elif bbox.right > 1 - margin_ths and bbox.left > 0.5:
            horizontal = "right"
        elif center_ths_low < center_x < center_ths_high:
            horizontal = "center"

        # Vertical positioning logic
        if center_y == min(all_centers_y) and bbox.bottom < 0.5:
            vertical = "topmost"
        elif center_y == max(all_centers_y) and bbox.top > 0.5:
            vertical = "bottommost"
        elif bbox.top < margin_ths and bbox.bottom < 0.5:
            vertical = "top"
        elif bbox.bottom > 1 - margin_ths and bbox.top > 0.5:
            vertical = "bottom"
        elif center_ths_low < center_y < center_ths_high:
            vertical = "center"

        description = None
        # Combine horizontal and vertical positions for the description
        if (horizontal, vertical) in [
            ("leftmost", "topmost"),
            ("leftmost", "top"),
            ("left", "topmost"),
        ]:
            description = "in the top left corner"
        elif (horizontal, vertical) in [
            ("rightmost", "topmost"),
            ("rightmost", "top"),
            ("right", "topmost"),
        ]:
            description = "in the top right corner"
        elif (horizontal, vertical) in [
            ("leftmost", "bottommost"),
            ("leftmost", "bottom"),
            ("left", "bottommost"),
        ]:
            description = "in the bottom left corner"
        elif (horizontal, vertical) in [
            ("rightmost", "bottommost"),
            ("rightmost", "bottom"),
            ("right", "bottommost"),
        ]:
            description = "in the bottom right corner"
        elif horizontal == "leftmost":
            description = "on the left side"
        elif horizontal == "rightmost":
            description = "on the right side"
        elif vertical == "topmost":
            description = "at the top"
        elif vertical == "bottommost":
            description = "at the bottom"
        elif horizontal == "center" and vertical == "center":
            description = "in the center"
        if description:
            description = f"{self.entity_info.text} is {description} of the composition"

        return description

    def get_edge_description(self, child_node):
        # Calculate the center of the current node's bbox
        self_center_x = (self.bbox.left + self.bbox.right) / 2
        self_center_y = (self.bbox.top + self.bbox.bottom) / 2

        # Calculate the center of the child node's bbox
        child_center_x = (child_node.bbox.left + child_node.bbox.right) / 2
        child_center_y = (child_node.bbox.top + child_node.bbox.bottom) / 2

        # Determine the relative direction
        dx = child_center_x - self_center_x
        dy = child_center_y - self_center_y

        # Determine the angle based on dx and dy
        angle = math.atan2(dy, dx) * (180 / math.pi)

        # Normalize the angle to [0, 360) range
        angle = angle % 360

        mode1 = np.random.random() < 0.4

        if 0 <= angle < 22.5 or 337.5 <= angle < 360:
            direction = "to the right of"
        elif 22.5 <= angle < 67.5:
            if mode1:
                direction = "to the bottom-right of"
            else:
                direction = "below and to the right of"
        elif 67.5 <= angle < 112.5:
            direction = "below"
        elif 112.5 <= angle < 157.5:
            if mode1:
                direction = "to the bottom-left of"
            else:
                direction = "below and to the left of"
        elif 157.5 <= angle < 202.5:
            direction = "to the left of"
        elif 202.5 <= angle < 247.5:
            if mode1:
                direction = "to the top-left of"
            else:
                direction = "above and to the left of"
        elif 247.5 <= angle < 292.5:
            direction = "above"
        elif 292.5 <= angle < 337.5:
            if mode1:
                direction = "to the top-right of"
            else:
                direction = "above and to the right of"

        # Construct the description
        description = (
            f"{child_node.entity_info.text} is {direction} {self.entity_info.text}."
        )

        return description

    @classmethod
    def build_mst(
        cls, entities: list[EntityInfo], abs_bboxes: list[Bbox], rel_bboxes: list[Bbox]
    ) -> "MSTNode":
        """
        Build a minimum spanning tree for the given entities and bounding boxes.

        Parameters
        ----------
        entities: list[EntityInfo]
            The list of entities.
        abs_bboxes: list[Bbox]
            The list of bounding boxes in absolute coordinates,
            i.e. with respect to the entire image.
        rel_bboxes: list[Bbox]
            The list of bounding boxes in relative coordinates,
            i.e. with respect to the cropped image.

        Returns
        -------
        MSTNode
            The root node of the minimum spanning tree.
        """
        # Compute centers of bboxes
        centers = [
            [(bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2]
            for bbox in abs_bboxes
        ]

        # Compute pairwise Euclidean distances between centers
        distance_matrix = cdist(centers, centers, metric="euclidean")

        # Compute the minimum spanning tree
        mst_sparse = minimum_spanning_tree(csr_matrix(distance_matrix))
        mst = mst_sparse.toarray()
        mst = np.maximum(mst, mst.T)

        # Convert the MST to MSTNode objects
        nodes = [
            cls(entity_info=entity, bbox=bbox, rel_bbox=rel_bbox)
            for entity, bbox, rel_bbox in zip(entities, abs_bboxes, rel_bboxes)
        ]

        # Create a tree structure from the MST adjacency matrix
        root_index = cls.choose_root_node(abs_bboxes)
        tree = cls.create_tree_from_mst(mst, nodes, root_index)

        return tree

    @staticmethod
    def create_tree_from_mst(
        mst: np.ndarray, nodes: list["MSTNode"], root_index: int
    ) -> "MSTNode":
        visited = set()

        def dfs(index: int) -> "MSTNode":
            visited.add(index)
            node = nodes[index]
            for i, edge_weight in enumerate(mst[index]):
                if (
                    edge_weight > 0 and i not in visited
                ):  # There's an edge and it's not visited
                    child_node = dfs(i)
                    node.add_child(child_node)
            return node

        return dfs(root_index)

    @staticmethod
    def choose_root_node(bboxes: list[Bbox]) -> int:
        centers = np.array(
            [
                [(bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2]
                for bbox in bboxes
            ]
        )
        x_coords, y_coords = centers[:, 0], centers[:, 1]

        # Define criteria for selecting root nodes
        criteria = {
            "leftmost": np.argmin(x_coords),
            "rightmost": np.argmax(x_coords),
            "topmost": np.argmin(y_coords),
            "bottommost": np.argmax(y_coords),
            "middle_x": np.argsort(x_coords)[len(x_coords) // 2],
            "middle_y": np.argsort(y_coords)[len(y_coords) // 2],
        }

        # Randomly choose a criterion and return the corresponding node index
        chosen_criterion = choice(list(criteria.keys()))
        return criteria[chosen_criterion]
