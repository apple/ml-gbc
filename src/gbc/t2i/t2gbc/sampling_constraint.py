# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from copy import deepcopy

import torch
from transformers import PreTrainedTokenizerBase


# Function to check if input_ids ends with certain tokens
def ends_with(input_ids, tokens):
    return len(input_ids) >= len(tokens) and input_ids[-len(tokens) :] == tokens


def create_prefix_allowed_tokens_fn(
    tokenizer: PreTrainedTokenizerBase,
    allow_composition: bool = True,
    star_graph: bool = False,
    entity_lists: list[list[str]] | None = None,
):

    eos_token_id = tokenizer.eos_token_id
    all_tokens = list(range(tokenizer.vocab_size))
    all_tokens_except_eos = [token for token in all_tokens if token != eos_token_id]

    # Precompute token IDs for efficiency
    type_tokens = tokenizer("\ntype:", add_special_tokens=False).input_ids[2:]
    entity_tokens = tokenizer("entity", add_special_tokens=False).input_ids
    composition_tokens = tokenizer("composition", add_special_tokens=False).input_ids
    is_leave_tokens = tokenizer("\nis_leave:", add_special_tokens=False).input_ids[2:]
    true_tokens = tokenizer("True", add_special_tokens=False).input_ids
    node_hash_tokens = tokenizer("\nNode #", add_special_tokens=False).input_ids[2:]

    # Keep track of word generation state
    generation_state = {
        "counter": 1,
        "current_tokens": None,
        "current_index": 0,
        "idx": 0,
    }
    if entity_lists is not None:
        entity_lists = deepcopy(entity_lists)

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor):

        if entity_lists is not None and batch_id < len(entity_lists):
            entity_list = entity_lists[batch_id]
        else:
            entity_list = []

        input_ids_list = input_ids.tolist()
        # Choose allowed tokens based on whether entity_list_copy is empty
        allowed_tokens = all_tokens_except_eos if entity_list else all_tokens

        # If in the middle of generating a word, only allow the next token
        if generation_state["current_tokens"] is not None:
            if generation_state["current_index"] + 1 < len(
                generation_state["current_tokens"]
            ):
                generation_state["current_index"] += 1
                next_token_id = generation_state["current_tokens"][
                    generation_state["current_index"]
                ]
                return [next_token_id]
            else:
                # Finished generating the word
                generation_state["current_tokens"] = None
                generation_state["current_index"] = -1

        # Deal with counter and entity list
        counter_str = str(generation_state["counter"])
        counter_tokens = tokenizer(
            "\n" + counter_str, add_special_tokens=False
        ).input_ids[2:]
        node_counter_tokens = node_hash_tokens + counter_tokens
        generation_state["idx"] += 1

        # After 'Node #', generate the counter number
        if ends_with(input_ids_list, node_hash_tokens):
            generation_state["current_tokens"] = counter_tokens
            generation_state["current_index"] = 0
            return [counter_tokens[0]]

        # After 'Node #n:', generate the next entity from entity_list_copy
        if ends_with(input_ids_list, node_counter_tokens):
            generation_state["counter"] += 1
            if entity_list:
                entity_name = " " + entity_list.pop(0) + "\n"
                entity_name_tokens = tokenizer(
                    entity_name, add_special_tokens=False
                ).input_ids
                generation_state["current_tokens"] = entity_name_tokens
                generation_state["current_index"] = 0
                return [entity_name_tokens[0]]

        # After 'type: ', allow 'entity' or 'composition' based on allow_composition
        if ends_with(input_ids_list, type_tokens):
            if allow_composition:
                # Allow 'entity' or 'composition'
                possible_first_tokens = [entity_tokens[0], composition_tokens[0]]
                generation_state["word_options"] = {
                    entity_tokens[0]: entity_tokens,
                    composition_tokens[0]: composition_tokens,
                }
                return possible_first_tokens
            else:
                # Only allow 'entity'
                generation_state["current_tokens"] = entity_tokens
                generation_state["current_index"] = 0
                return [entity_tokens[0]]

        # Handle with is_leave
        if ends_with(input_ids_list, is_leave_tokens) and star_graph:
            generation_state["current_tokens"] = true_tokens
            generation_state["current_index"] = 0
            return [true_tokens[0]]

        # Handle the case where 'entity' or 'composition' is being generated
        if "word_options" in generation_state:
            last_token = input_ids_list[-1]
            if last_token in generation_state["word_options"]:
                # The model has chosen a word to generate
                chosen_tokens = generation_state["word_options"][last_token]
                generation_state["current_tokens"] = chosen_tokens
                generation_state["current_index"] = 1
                del generation_state["word_options"]
                return [chosen_tokens[1]] if len(chosen_tokens) > 1 else allowed_tokens
            else:
                # Model generated an unexpected token; reset state
                generation_state["current_tokens"] = None
                generation_state["current_index"] = -1
                del generation_state["word_options"]
                return allowed_tokens

        return allowed_tokens

    return prefix_allowed_tokens_fn
