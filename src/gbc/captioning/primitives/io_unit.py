# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import re
from typing import Optional, Literal
from pydantic import BaseModel


from gbc.data import Description


__all__ = ["EntityInfo", "QueryResult", "RefPosition", "find_ref_poss"]


class RefPosition(BaseModel):
    """
    Represents a reference to a specific segment within a list of text descriptions.

    Attributes
    ----------
    desc_index : int
        Index of the target description within the list.
    start : int
        Start position of the text segment within the target description.
    end : int
        End position of the text segment within the target description.
    """

    desc_index: int
    start: int
    end: int


def find_ref_poss(desc_texts: list[str], entity_text: str) -> list[RefPosition]:
    """
    Find all positions of the entity text within the provided description texts.

    Parameters
    ----------
    desc_texts
        List of description texts to search within.
    entity_text
        The entity text to find within the description texts.

    Returns
    -------
    list[RefPosition]
        List of reference positions where the entity text occurs in the
        description texts.
    """
    ref_poss = []
    # Create a pattern that matches the entity text only
    # if it's surrounded by word boundaries
    pattern = r"\b" + re.escape(entity_text) + r"\b"

    for idx, desc_text in enumerate(desc_texts):
        # Find all occurrences of the entity text in the description text,
        # considering word boundaries
        for match in re.finditer(pattern, desc_text, re.IGNORECASE):
            start = match.start()
            end = match.end()
            ref_pos = RefPosition(desc_index=idx, start=start, end=end)
            ref_poss.append(ref_pos)

    return ref_poss


class EntityInfo(BaseModel):
    """
    Stores information about entities including their label and text.

    Attributes
    ----------
    label : Literal["image", "entity", "single", "multiple", "relation", "composition"]
        Label indicating the type of entity.

        - In ``action_input.entity_info``, this translates to node label.
        - In ``action_input.entities``, the label ``single`` and ``multiple`` affect
          how detected bounding boxes are post processed.
    text : str | None
        Text associated with the entity.
        The texts from ``entities`` in
        :class:`~gbc.captioning.primitives.io_unit.QueryResult`
        are later parsed as edge labels.
    entity_id : str
        Identifier for the entity.
    """

    label: Literal["image", "entity", "single", "multiple", "relation", "composition"]
    text: Optional[str] = None
    entity_id: str = ""


class QueryResult(BaseModel):
    """
    Represents the result of a query, including descriptions and entities.

    Attributes
    ----------
    descs : list of tuple of (Description, list of str)
        List of descriptions and their associated entity ids.
        This second part is only useful for relation queries, in which case
        it is used to indicate which entities are involved in the relation.
    entities : list of tuple of (EntityInfo, list of RefPosition)
        List of entities and their reference positions within the descriptions.
    raw : str | None
        Raw query result data.
    """

    descs: list[tuple[Description, list[str]]]
    entities: list[tuple[EntityInfo, list[RefPosition]]]
    raw: Optional[str] = None
