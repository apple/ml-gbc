# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import traceback
from typing import Optional, Union
from PIL import Image

from gbc.utils import get_gbc_logger

from .llava_base import load_llava_model, llava_query_single
from ..query_prototype import (
    MllmQueryPrototype,
    ImageQuery,
    EntityQuery,
    RelationQuery,
    CompositionQuery,
)
from ...primitives import QueryResult


__all__ = [
    "LlavaQueryAction",
    "LlavaImageQuery",
    "LlavaEntityQuery",
    "LlavaCompositionQuery",
    "LlavaRelationQuery",
]


class LlavaQueryAction(MllmQueryPrototype):
    """
    Base class for all LLaVA query actions.
    """

    def load_model(self, **kwargs):
        return load_llava_model(**kwargs)

    def query_prelim(
        self,
        image: Image.Image,
        filled_in_query: Optional[Union[str, list[str], tuple[str]]] = None,
    ) -> QueryResult:
        try:
            query_output = llava_query_single(
                self.query_model,
                image,
                self.query_message,
                filled_in_query=filled_in_query,
                system_message=self.system_message,
                temperature=self.query_kwargs.pop("temperature", 0.1),
                **self.query_kwargs,
            )
        except Exception as e:
            logger = get_gbc_logger()
            logger.warning(f"Failed to query: {e}")
            traceback.print_exc()
            query_output = ""
        return query_output


class LlavaImageQuery(ImageQuery, LlavaQueryAction):
    pass


class LlavaEntityQuery(EntityQuery, LlavaQueryAction):
    pass


class LlavaRelationQuery(RelationQuery, LlavaQueryAction):
    pass


class LlavaCompositionQuery(CompositionQuery, LlavaQueryAction):
    pass
