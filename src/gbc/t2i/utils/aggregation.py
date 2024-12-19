# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import torch


def aggregate_embeddings(
    embeddings: torch.Tensor, n_elements: list[int], mode: str, **kwargs
):
    if mode == "concat":
        return concat_aggregate_embeddings_vectorize(embeddings, n_elements, **kwargs)
    if mode == "first":
        return first_aggregate_embeddings(embeddings, n_elements, **kwargs)
    if mode == "mean":
        return mean_aggregate_embeddings(embeddings, n_elements, **kwargs)
    raise ValueError(f'Invalid aggregation mode "{mode}"')


def concat_aggregate_embeddings(
    embeddings: torch.Tensor,
    n_elements: list[int],
    pad_value: float = 0,
    pad_to_n_elements: int | None = None,
):
    assert sum(n_elements) == len(embeddings)

    max_n_elements = pad_to_n_elements or max(n_elements)
    batch_size = len(n_elements)
    sequence_length = embeddings.shape[1]
    cat_embeddings = (
        embeddings.new_ones(
            (batch_size, max_n_elements * sequence_length, *embeddings.shape[2:])
        )
        * pad_value
    )
    start_idx = 0

    for b, n in enumerate(n_elements):
        cat_embeddings[b, 0 : n * sequence_length] = embeddings[
            start_idx : start_idx + n
        ].flatten(end_dim=1)
        start_idx += n
    return cat_embeddings


def get_batch_and_position_indices_for_concat_aggregate(
    n_elements: torch.Tensor,
    sequence_length: int,
):
    assert n_elements.dtype == torch.long
    batch_size = n_elements.size(0)
    n_seq_per_batch = n_elements * sequence_length
    max_n_seq = n_seq_per_batch.max()

    # Create positions and masks
    positions_per_batch = torch.arange(max_n_seq).unsqueeze(0).to(n_elements.device)
    positions_per_batch = positions_per_batch.repeat(batch_size, 1)
    mask = positions_per_batch < n_seq_per_batch.unsqueeze(1)
    positions_flat = positions_per_batch[mask]

    # Generate batch indices and positions
    batch_indices_flat = torch.arange(batch_size).to(n_elements.device)
    batch_indices_flat = batch_indices_flat.unsqueeze(1).repeat(1, max_n_seq)[mask]
    return batch_indices_flat, positions_flat


# @profile
def concat_aggregate_embeddings_vectorize(
    embeddings: torch.Tensor,
    n_elements: list[int] | torch.LongTensor,
    pad_value: float = 0,
    pad_to_n_elements: int | None = None,
    # Needs to be precomputed for efficiency
    batch_indices_flat: torch.LongTensor | None = None,
    positions_flat: torch.LongTensor | None = None,
    cat_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    # This would cause the function to be slower
    # assert torch.sum(n_elements) == len(embeddings)

    sequence_length = embeddings.shape[1]

    if not isinstance(n_elements, torch.Tensor):
        n_elements = torch.tensor(n_elements)

    if batch_indices_flat is None or positions_flat is None:
        (
            batch_indices_flat,
            positions_flat,
        ) = get_batch_and_position_indices_for_concat_aggregate(
            n_elements,
            sequence_length,
        )

    # Need pad_to_n_elements to create cat_embeddings in case of padding
    max_n_elements = pad_to_n_elements or n_elements.max()
    assert max_n_elements >= n_elements.max(), (
        f"pad_to_n_elements is {pad_to_n_elements}"
        f"but max_n_elements is {n_elements.max()}"
    )
    max_n_seq = max_n_elements * sequence_length

    # Flatten embeddings along the first two dimensions
    embeddings_flat = embeddings.flatten(0, 1)

    # Initialize the output tensor with the pad_value
    if cat_embeddings is None:
        batch_size = n_elements.shape[0]
        cat_embeddings = embeddings.new_full(
            (batch_size, max_n_seq, *embeddings.shape[2:]), pad_value
        )

    cat_embeddings[batch_indices_flat, positions_flat] = embeddings_flat
    # print(cat_embeddings.shape)

    return cat_embeddings


def split_aggregate_embeddings(
    cat_embeddings: torch.Tensor,
    n_elements: list[int] | torch.Tensor,
    sequence_length: int,
):
    """
    Reverses the concat_aggregate_embeddings function by splitting the concatenated
    embeddings back into their original variable-sized batches.

    Parameters
    ----------
        cat_embeddings
            The concatenated and padded embeddings tensor of shape
            [batch_size, max_n_elements * sequence_length, *embedding_dims]
        n_elements
            A list of integers indicating the number of elements in each batch.
        sequence_length
            The original sequence length of the embeddings before concatenation.

    Returns:
        torch.Tensor
            The reconstructed embeddings tensor of shape
            (sum(n_elements), sequence_length, *embedding_dims).
    """
    batch_size, max_total_length, *embedding_dims = cat_embeddings.shape
    device = cat_embeddings.device

    if isinstance(n_elements, torch.Tensor):
        n_elements_tensor = n_elements.to(cat_embeddings.device)
    else:
        n_elements_tensor = torch.tensor(n_elements, device=cat_embeddings.device)
    valid_lengths = n_elements_tensor * sequence_length  # Shape: (batch_size,)

    # Create a mask for valid positions
    positions = (
        torch.arange(max_total_length, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    valid_mask = positions < valid_lengths.unsqueeze(
        1
    )  # Shape: (batch_size, max_total_length)

    # Compute flat indices for valid positions
    flat_positions = torch.arange(batch_size * max_total_length, device=device)
    valid_indices = flat_positions[
        valid_mask.reshape(-1)
    ]  # Shape: (total_valid_positions,)

    # Extract valid embeddings
    cat_embeddings_flat = cat_embeddings.reshape(-1, *embedding_dims)
    valid_embeddings_flat = cat_embeddings_flat[
        valid_indices
    ]  # Shape: (total_valid_positions, ...)

    # Reshape back to (sum(n_elements), sequence_length, *embedding_dims)
    total_elements = n_elements_tensor.sum()
    reconstructed_embeddings = valid_embeddings_flat.reshape(
        total_elements, sequence_length, *embedding_dims
    )

    return reconstructed_embeddings


def first_aggregate_embeddings(
    embeddings: torch.Tensor,
    n_elements: list[int],
    n_generated_per_image: list[int] | None = None,
):
    assert sum(n_elements) == len(embeddings)
    if n_generated_per_image is None:
        n_generated_per_image = [1] * len(n_elements)
    assert len(n_generated_per_image) == len(n_elements)

    batch_size = sum(n_generated_per_image)
    cat_embeddings = embeddings.new_zeros((batch_size, *embeddings.shape[1:]))

    offset = 0
    start_idx = 0
    for n_generated, n_all in zip(n_generated_per_image, n_elements):
        cat_embeddings[offset : offset + n_generated] = embeddings[
            start_idx : start_idx + n_generated
        ]
        offset += n_generated
        start_idx += n_all
    return cat_embeddings


def mean_aggregate_embeddings(embeddings: torch.Tensor, n_elements: list[int]):
    assert sum(n_elements) == len(embeddings)

    batch_size = len(n_elements)
    cat_embeddings = embeddings.new_zeros((batch_size, *embeddings.shape[1:]))
    start_idx = 0

    for b, n in enumerate(n_elements):
        cat_embeddings[b] = embeddings[start_idx : start_idx + n].mean(dim=0)
        start_idx += n
    return cat_embeddings


if __name__ == "__main__":

    n_elements = [2, 3, 1]
    embeddings = torch.randn(6, 4, 5)
    print(embeddings)
    print(concat_aggregate_embeddings(embeddings=embeddings, n_elements=n_elements))
    print(
        concat_aggregate_embeddings_vectorize(
            embeddings=embeddings, n_elements=n_elements
        )
    )
    print(first_aggregate_embeddings(embeddings=embeddings, n_elements=n_elements))

    cat_embeddings = concat_aggregate_embeddings_vectorize(embeddings, n_elements)

    reconstructed_embeddings = split_aggregate_embeddings(
        cat_embeddings=cat_embeddings, n_elements=n_elements, sequence_length=4
    )
    print(reconstructed_embeddings)
