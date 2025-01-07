# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import torch

from .basics import plural_to_singular
from .classifiers import load_emb_classifier, load_emb_pair_classifier


def potential_same_object(text1: str, text2: str, config=None) -> bool:
    singular_text1 = plural_to_singular(text1)
    singular_text2 = plural_to_singular(text2)
    if singular_text1 == singular_text2:
        return True
    if config is None:
        return False
    model = load_emb_pair_classifier(
        config.text_pair_model_path, gpu_id=getattr(config, "gpu_id", 0)
    )
    with torch.no_grad():
        output1 = model((text1, text2)).squeeze().cpu().numpy()
        output2 = model((singular_text1, singular_text2)).squeeze().cpu().numpy()
    return (output1 > 0) or (output2 > 0)


def suitable_for_detection(text: str, config=None) -> bool:
    if text in ["right", "left", "top", "bottom", "front", "back", "side"]:
        return False
    # Detection with digit would cause collision between entity and composition nodes
    if text.isdigit():
        return False
    # This will break the naming convention of the node ids
    # As we split each detection result by '_'
    if "_" in text:
        return False
    # Otherwise we return True if no model is loaded
    if config is None:
        return True
    model = load_emb_classifier(
        config.text_binary_model_path, gpu_id=getattr(config, "gpu_id", 0)
    )
    with torch.no_grad():
        output = model(text).squeeze().cpu().numpy()
    return output > 0
