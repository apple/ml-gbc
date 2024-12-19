# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""
This module defines classes for automated selection and instantiation of actions.

All classes expect a configuration dictionary with a specific structure in input
and instantiate the corresponding action with `Hydra <https://hydra.cc/>`_.
A default action is used if no configuration is provided or if no corresponding
field is found in the configuration.

The following example shows how to use an action defined in this module.

.. code-block:: python

   action = AutoDetectionActionFromImage(config)
"""

from typing import Callable, Optional
from omegaconf import DictConfig
from hydra.utils import instantiate

from gbc.utils import get_gbc_logger
from .primitives.action import Action


__all__ = [
    "AutoImageQuery",
    "AutoEntityQuery",
    "AutoRelationQuery",
    "AutoCompositionQuery",
    "AutoDetectionActionFromImage",
    "AutoDetectionActionFromEntity",
]


class AutoAction(object):

    _cache: dict[tuple, "AutoAction"] = dict()

    def __new__(
        cls,
        candidate_names: list[str],
        get_default_action: Callable[[], Action],
        config: Optional[DictConfig] = None,
    ):
        cache_key = (tuple(candidate_names), config)

        if cache_key in cls._cache:
            return cls._cache[cache_key]

        logger = get_gbc_logger()
        logger.info(f"Initialize {cls.__name__}")

        if config is not None:
            queries = config.get("queries", {})
            for name in candidate_names:
                if name in queries:
                    instance = instantiate(queries[name])
                    cls._cache[cache_key] = instance
                    return instance

        default_action = get_default_action()
        cls._cache[cache_key] = default_action
        return default_action


class AutoDetectionActionFromImage(AutoAction):
    """
    Automates creation of detection actions from image node.
    It reads from either the ``detection_from_image`` or ``detection`` field
    under the ``queries`` section in the configuration.

    Expected input: :class:`~gbc.captioning.primitives.action_io.ActionInput`.

    The default action is a
    :class:`~gbc.captioning.detection.detection_action.DetectionAction`
    with no ``max_rel_area`` specified.
    """

    def __new__(cls, config: Optional[DictConfig] = None):

        candidate_names = ["detection_from_image", "detection"]

        def get_default_action():
            from .detection.detection_action import DetectionAction

            return DetectionAction()

        return super().__new__(cls, candidate_names, get_default_action, config)


class AutoDetectionActionFromEntity(AutoAction):
    """
    Automates creation of detection actions from entity node.
    It reads from either the ``detection_from_entity`` or ``detection`` field
    under the ``queries`` section in the configuration.

    Expected input:
    :class:`~gbc.captioning.primitives.action_io.ActionInputWithEntities`.

    The default action is a
    :class:`~gbc.captioning.detection.detection_action.DetectionAction`
    with ``max_rel_area=0.8``.
    """

    def __new__(cls, config: Optional[DictConfig] = None):
        candidate_names = ["detection_from_entity", "detection"]

        def get_default_action():
            from .detection.detection_action import DetectionAction

            return DetectionAction(max_rel_area=0.8)

        return super().__new__(cls, candidate_names, get_default_action, config)


class AutoImageQuery(AutoAction):
    """
    Automates creation of image queries. It reads from either the ``image``
    or ``image_query`` field under the ``queries`` section in the configuration.

    Expected input: :class:`~gbc.captioning.primitives.action_io.ActionInput`.

    The default action is a
    :class:`~gbc.captioning.pixtral.PixtralImageQuery`.
    """

    def __new__(cls, config: Optional[DictConfig] = None):
        candidate_names = ["image", "image_query"]

        def get_default_action():
            from .mllm.pixtral.pixtral_queries import PixtralImageQuery

            system_file = "prompts/captioning/system_image.txt"
            query_file = "prompts/captioning/query_image.txt"
            return PixtralImageQuery(system_file, query_file)

        return super().__new__(cls, candidate_names, get_default_action, config)


class AutoEntityQuery(AutoAction):
    """
    Automates creation of entity queries. It reads from either the ``entity``
    or ``entity_query`` field under the ``queries`` section in the configuration.

    Expected input:
    :class:`~gbc.captioning.primitives.action_io.ActionInputWithEntities`.

    The default action is a
    :class:`~gbc.captioning.pixtral.PixtralEntityQuery`.
    """

    def __new__(cls, config: Optional[DictConfig] = None):
        candidate_names = ["entity", "entity_query"]

        def get_default_action():
            from .mllm.pixtral.pixtral_queries import PixtralEntityQuery

            system_file = "prompts/captioning/system_entity.txt"
            query_file = "prompts/captioning/query_entity.txt"
            return PixtralEntityQuery(system_file, query_file)

        return super().__new__(cls, candidate_names, get_default_action, config)


class AutoRelationQuery(AutoAction):
    """
    Automates creation of relation queries. It reads from either the ``relation``
    or ``relation_query`` field under the ``queries`` section in the configuration.

    Expected input:
    :class:`~gbc.captioning.primitives.action_io.ActionInputWithEntities`.

    The default action is a
    :class:`~gbc.captioning.pixtral.PixtralRelationQuery`.
    """

    def __new__(cls, config: Optional[DictConfig] = None):
        candidate_names = ["relation", "relation_query"]

        def get_default_action():
            from .mllm.pixtral.pixtral_queries import PixtralRelationQuery

            system_file = "prompts/captioning/system_relation.txt"
            query_file = "prompts/captioning/query_relation.txt"
            return PixtralRelationQuery(system_file, query_file)

        return super().__new__(cls, candidate_names, get_default_action, config)


class AutoCompositionQuery(AutoAction):
    """
    Automates creation of composition queries. It reads from either the ``composition``
    or ``composition_query`` field under the ``queries`` section in the configuration.

    Expected input:
    :class:`~gbc.captioning.primitives.action_io.ActionInputWithBboxes`.

    The default action is a
    :class:`~gbc.captioning.pixtral.PixtralCompositionQuery`.
    """

    def __new__(cls, config: Optional[DictConfig] = None):
        candidate_names = ["composition", "composition_query"]

        def get_default_action():
            from .mllm.pixtral.pixtral_queries import PixtralCompositionQuery

            system_file = "prompts/captioning/system_composition.txt"
            query_file = "prompts/captioning/query_composition.txt"
            return PixtralCompositionQuery(system_file, query_file)

        return super().__new__(cls, candidate_names, get_default_action, config)
