# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import torch
from tqdm import tqdm
from functools import cache
from detoxify import Detoxify

from gbc.data.graph import GbcGraph, GbcGraphFull


def compute_toxicity_scores(
    gbc_graphs: list[GbcGraph], model_name: str = "original", device: str | None = None
) -> list[GbcGraphFull]:
    """
    Compute toxicity scores for GBC graphs.

    This function calculates toxicity scores for all captions associated with
    vertices in the provided GBC graphs.
    The scores are computed using the specified toxicity model.

    Parameters
    ----------
    gbc_graphs
        A list of GBC graphs for which toxicity scores need to be computed.
        If a :class:`~GbcGraph` is passed,
        a new :class:`~GbcGraphFull` instance is created.
    model_name
        The name of the toxicity model to use for computing scores.
        Default is ``original``.
    device
        The device to run the toxicity model on (e.g., ``cpu`` or ``cuda``).

    Returns
    -------
    list of GbcGraphFull
        A list of :class:`~GbcGraphFull` objects with updated toxicity scores
        for captions.

    Notes
    -----
    - If an input ``gbc_graph`` is already in :class:`~GbcGraphFull`,
      the modifications will be in-place.
    - If a caption already has toxicity scores computed, they are retained.
    """

    gbc_graphs_with_toxicity_scores = []

    for gbc_graph in tqdm(gbc_graphs, desc="Computing toxicity scores"):
        if not isinstance(gbc_graph, GbcGraphFull):
            gbc_graph = GbcGraphFull.from_gbc_graph(gbc_graph)
        for vertex in gbc_graph.vertices:
            for caption in vertex.descs:
                if caption.toxicity_scores is None or len(caption.toxicity_scores) == 0:
                    caption.toxicity_scores = compute_toxicity(
                        caption.text, model_name, device
                    )
        gbc_graphs_with_toxicity_scores.append(gbc_graph)

    return gbc_graphs_with_toxicity_scores


@cache
def get_toxicity_model(model_name="original", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Detoxify(model_name, device=device)
    return model


def compute_toxicity(caption: str, model_name="original", device=None):
    model = get_toxicity_model(model_name, device)
    scores = model.predict(caption)
    # from float32 to float
    scores = {key: float(value) for key, value in scores.items()}
    return scores
