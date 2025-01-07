# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class GraphAttnMetaPerImage:
    """
    Attributes
    ----------
    adjacency_with_slice_mapping: list[list[tuple[int, list[slice]]]]
        - A list where each element corresponds to a vertex (node) in a graph.
        - Each element is a list of tuples representing edges from the vertex
          to its targets.
        - Each tuple is of the form (target_vertex_id, list_of_slices), where:
            - target_vertex_id (int): The ID of the target vertex this edge
              points to.
            - list_of_slices (List[slice]): A list of slice objects indicating
              index ranges to map.
    caption_mask: torch.BoolTensor
        - A tensor representing the attention mask for the caption.
        - Shape: (num_captions, sequence_length)
    sequence_length: int
        The context length of each encoded caption.
    """

    adjacency_with_slice_mapping: list[list[tuple[int, list[slice]]]]
    caption_mask: torch.BoolTensor
    sequence_length: int

    def __len__(self):
        return len(self.adjacency_with_slice_mapping)

    def get_adjacency(self) -> list[list[int]]:
        adjacency = []
        for children_with_slice_mapping_i in self.adjacency_with_slice_mapping:
            adjacency.append([child for child, _ in children_with_slice_mapping_i])
        return adjacency

    @classmethod
    def from_prompts_and_adjacency(
        cls,
        prompts: list[str],
        labeled_adjacency: list[list[tuple[int, list[str]]]],
        tokenizer: PreTrainedTokenizerBase,
    ):
        caption_input_ids = []
        caption_mask = torch.ones(
            len(prompts), tokenizer.model_max_length, dtype=torch.bool
        )

        for idx, caption in enumerate(prompts):
            tokenizer_outputs = tokenizer(
                caption, padding="max_length", return_tensors="pt", truncation=True
            )
            # Shape (1, model_max_length)
            input_ids = tokenizer_outputs["input_ids"].flatten().tolist()
            attention_mask = tokenizer_outputs["attention_mask"].flatten()
            final_idx = attention_mask.cumsum(dim=0).argmax(dim=0).item()
            input_ids = input_ids[: final_idx + 1]
            caption_input_ids.append(input_ids)

        converted_labeled_adjacency = []

        for prompt_idx, (parent_tokens, adjacency) in enumerate(
            zip(caption_input_ids, labeled_adjacency)
        ):
            converted = []
            for target, labels in adjacency:
                mapping_slices = []
                for label in labels:
                    label_tokens = tokenizer.encode(label)[1:-1]
                    n_label_tokens = len(label_tokens)
                    token_start = 1
                    while token_start < len(parent_tokens) - n_label_tokens:
                        if (
                            parent_tokens[token_start : token_start + n_label_tokens]
                            == label_tokens
                        ):
                            mapping_slices.append(
                                slice(token_start, token_start + n_label_tokens)
                            )
                            caption_mask[
                                prompt_idx, token_start : token_start + n_label_tokens
                            ] = False
                            token_start += n_label_tokens
                        else:
                            token_start += 1
                # None target is for caption mask only
                if len(mapping_slices) > 0 and target is not None:
                    converted.append((target, mapping_slices))
            converted_labeled_adjacency.append(converted)

        return cls(
            adjacency_with_slice_mapping=converted_labeled_adjacency,
            sequence_length=tokenizer.model_max_length,
            caption_mask=caption_mask,
        )

    def to_k_index_mapping(self) -> list[list[int]]:
        """
        Converts an adjacency list with slice mappings into an index mapping
        that maps vertex to token position that map to it after prompts
        are concatenated.
        """
        # Initialize the index mapping list for each vertex
        index_mapping = [[] for _ in range(len(self.adjacency_with_slice_mapping))]
        # Loop through all vertices
        for vertex_id in range(len(self.adjacency_with_slice_mapping)):
            # Iterate over all vertices to find edges pointing to the current vertex
            for potential_parent, edges in enumerate(self.adjacency_with_slice_mapping):
                # Iterate over edges from potential_parent
                for target, slices in edges:
                    # Check if the edge points to the current vertex
                    if target == vertex_id:
                        # Iterate over each slice in the edge
                        for slc in slices:
                            # Collect indices from the slice,
                            # adjusted by parent's unit seq
                            indices = [
                                i + potential_parent * self.sequence_length
                                for i in range(slc.start, slc.stop)
                            ]
                            index_mapping[vertex_id].extend(indices)
        return index_mapping


if __name__ == "__main__":

    from transformers import AutoTokenizer

    prompts = ["A banana on table.", "The banana is red."]
    labeled_adjacency = [[(1, ["banana"])], []]
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    graph_attn_meta = GraphAttnMetaPerImage.from_prompts_and_adjacency(
        prompts, labeled_adjacency, tokenizer
    )
    print(graph_attn_meta)
