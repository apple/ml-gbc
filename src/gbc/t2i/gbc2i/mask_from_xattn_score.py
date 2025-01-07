# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import math

import torch

from gbc.utils import get_gbc_logger
from gbc.t2i.utils.segmentation import felzenszwalb_segmentation_intersection, otsu_1d
from gbc.t2i.modules.graph_attn import GraphAttnMetaPerImage


def get_mask_from_xattn_score(
    scores: torch.Tensor,
    graph_attn_meta=list[GraphAttnMetaPerImage | None],
    unit_seq_len: int = 77,
    n_generated_per_image: torch.Tensor | list[int] | None = None,
):
    """
    scores: shape [batch_size, n_heads, q_length, k_length]
    """
    if n_generated_per_image is None:
        n_generated_per_image = [1] * len(graph_attn_meta)
    elif isinstance(n_generated_per_image, torch.Tensor):
        n_generated_per_image = n_generated_per_image.tolist()
    b, h, q, k = scores.shape
    assert (
        sum(n_generated_per_image) == b
    ), f"{sum(n_generated_per_image)} != batch size {b}"

    max_n_captions_per_image = k // unit_seq_len

    # Selected caption would be marked as True
    selection_mask = torch.zeros(
        b, q, max_n_captions_per_image, dtype=torch.bool, device=scores.device
    )

    batch_index = 0

    # Select graph by graph
    for graph_idx, (num_generated, graph_attn_meta_i) in enumerate(
        zip(n_generated_per_image, graph_attn_meta)
    ):
        if graph_attn_meta_i is None:
            # attend to respective caption
            selection_mask[
                torch.arange(batch_index, batch_index + num_generated),
                :,
                torch.arange(num_generated),
            ] = True
            batch_index += num_generated
            continue

        # list[list[int]]
        adjacency = graph_attn_meta_i.get_adjacency()
        # list[list[int]]
        k_index_mapping_per_graph = graph_attn_meta_i.to_k_index_mapping()

        for caption_idx in range(num_generated):

            # By default assigned all to relevant caption
            selected_caption = torch.ones(q, dtype=torch.long) * caption_idx

            # We assume range(1, n_vertices) is in topological order
            # This is why selected_caption, and hence to_assign, would be
            # updated in the right order
            for vertex_id, children in enumerate(adjacency):
                to_assign = selected_caption == vertex_id
                if to_assign.sum() == 0 or len(children) == 0:
                    continue
                relevant_scores = torch.zeros(q, len(children) + 1)
                avg_scores = (
                    scores[
                        batch_index,
                        :,
                        :,
                        vertex_id * unit_seq_len : (vertex_id + 1) * unit_seq_len,
                    ]
                    .mean(dim=-1)  # Average across text tokens
                    .mean(dim=0)  # Average across head dimension
                )
                relevant_scores[:, 0] = avg_scores
                for pos, child in enumerate(children):
                    k_indices = k_index_mapping_per_graph[child]
                    # Compute average scores over h and k_indices
                    # Shape of avg_scores: [q]
                    avg_scores = (
                        scores[batch_index, :, :, k_indices].mean(dim=-1).mean(dim=0)
                    )
                    relevant_scores[:, pos + 1] = avg_scores
                selected_caption = (
                    assign_caption_based_on_scores_with_segmentation_and_otsu(
                        relevant_scores,
                        to_assign,
                        class_labels=[vertex_id] + children,
                        selected_caption=selected_caption.clone(),
                    )
                )

            # for selected_caption in selected_caption_history:
            selected_caption_indices = selected_caption.long()  # Shape: [q]

            # Assign to selection_mask for the current batch_index
            q_indices = torch.arange(q)
            selection_mask[batch_index, q_indices, selected_caption_indices] = True
            # print(selection_mask)

            batch_index += 1

    return selection_mask


def assign_caption_based_on_scores_with_segmentation_and_otsu(
    relevant_scores: torch.Tensor,  # Shape: [q, n_classes]
    to_assign: torch.BoolTensor,  # Shape: [q]
    class_labels: list[int],  # Length: n_classes
    selected_caption: torch.Tensor,  # Shape: [q], current assignments, in-place modif
) -> torch.Tensor:

    # Indices to assign
    indices_to_assign = torch.nonzero(to_assign, as_tuple=False).squeeze()
    # Ensure indices_to_assign is 1D
    if indices_to_assign.dim() == 0:
        indices_to_assign = indices_to_assign.unsqueeze(0)
    if indices_to_assign.numel() == 0:
        # Nothing to assign
        return selected_caption

    q_length, n_classes = relevant_scores.size()
    # Assume image is square
    side_length = int(math.sqrt(q_length))
    score_image = (
        relevant_scores.view(side_length, side_length, n_classes).cpu().numpy()
    )
    mask = to_assign.view(side_length, side_length).cpu().numpy()
    cluster_labels = felzenszwalb_segmentation_intersection(
        image=score_image[..., 1:], mask=mask
    )
    n_segmentation = cluster_labels.max() + 1
    # print("n_segmentation", n_segmentation)

    # Compute average scores per segmentation map
    scores_to_assign = torch.zeros(n_segmentation, n_classes)
    mask_per_cluster = torch.zeros(n_segmentation, q_length, dtype=bool)
    for cluster_label in range(n_segmentation):
        to_assign_cluster_indices = indices_to_assign[cluster_labels == cluster_label]
        mask_per_cluster[cluster_label, to_assign_cluster_indices] = True
        scores_to_assign[cluster_label] = relevant_scores[
            to_assign_cluster_indices
        ].mean(0)
    scores_to_assign_min = scores_to_assign.min(0, keepdim=True).values
    scores_to_assign_max = scores_to_assign.max(0, keepdim=True).values
    scores_to_assign = (scores_to_assign - scores_to_assign_min) / torch.clamp(
        (scores_to_assign_max - scores_to_assign_min), min=1e-6
    )
    # scores_to_assign_relative = scores_to_assign.argsort(dim=0).argsort(dim=0).float()

    # Initial assignment using argmax
    class_assignments = torch.zeros(n_segmentation, dtype=torch.long)
    current_scores = torch.full((n_segmentation,), float("-inf"))

    if n_segmentation == 1:
        logger = get_gbc_logger()
        logger.info("Only one segmentation")

    else:
        otsu_thresholds = torch.zeros(n_classes)
        for c in range(1, n_classes):
            otsu_thresholds[c] = otsu_1d(scores_to_assign[:, c])
            class_indices_mask = torch.logical_and(
                scores_to_assign[:, c] > otsu_thresholds[c],
                scores_to_assign[:, c] > current_scores,
            )
            class_assignments[class_indices_mask] = c
            current_scores[class_indices_mask] = scores_to_assign[class_indices_mask, c]

    # Map back to original class labels and selected caption tensor
    if not isinstance(class_labels, torch.Tensor):
        class_labels = torch.tensor(class_labels).to(device=relevant_scores.device)
    class_assignments = class_labels[class_assignments]

    for cluster_label in range(n_segmentation):
        cluster_indices = mask_per_cluster[cluster_label]
        selected_caption[cluster_indices] = class_assignments[cluster_label]

    return selected_caption


if __name__ == "__main__":

    q = 16
    n_classes = 3
    q_retain_proportion = 0.2
    device = torch.device("cpu")

    # Generate random scores
    torch.manual_seed(3)  # For reproducibility
    relevant_scores = torch.rand(q, n_classes, device=device)

    # Define to_assign with half True
    to_assign = torch.zeros(q, dtype=torch.bool, device=device)
    to_assign[:14] = True  # First half is True

    # Shuffle to_assign indices
    perm = torch.randperm(q)
    to_assign = to_assign[perm]

    # Define class labels
    class_labels = [100, 101, 102]

    # Initialize selected_caption
    selected_caption = torch.full(
        (q,), -1, dtype=torch.long, device=device
    )  # -1 indicates unassigned

    print("scores:", relevant_scores)
    print("initial selection:", selected_caption)
    print("to assign:", to_assign)

    print("----------------- Segmentation-based assignment -------------------")
    # Initialize selected_caption
    selected_caption = torch.full(
        (q,), -1, dtype=torch.long, device=device
    )  # -1 indicates unassigned

    # Call the function
    updated_selected_caption = (
        assign_caption_based_on_scores_with_segmentation_and_otsu(
            relevant_scores,
            to_assign,
            class_labels,
            selected_caption.clone(),
        )
    )
    print("assigned selected_caption:", updated_selected_caption)
