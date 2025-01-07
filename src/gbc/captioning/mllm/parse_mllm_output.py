# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import re
from typing import Optional, Set

from gbc.data import Description
from gbc.texts import plural_to_singular

from ..primitives.io_unit import EntityInfo, QueryResult, find_ref_poss
from ..primitives.action_io import (
    ActionInput,
    ActionInputWithEntities,
    ActionInputWithBboxes,
)


"""
Parse reply for Image query
"""


def parse_image(
    text, action_input: Optional[ActionInput] = None, max_n_features: int = 10
) -> QueryResult:
    lines = text.split("\n")  # Split the text into lines
    detailed_caption = ""
    concise_caption = ""
    mode = None  # Track the current section

    if action_input is None:
        node_id = ""
    else:
        node_id = action_input.first_entity_id

    processed_texts = set()
    identified_entities = []

    for line in lines:
        normalized_line = line.lstrip("#").strip()
        if normalized_line.startswith("Detailed Caption:"):
            mode = "detailed"
            detailed_caption += (
                normalized_line[len("Detailed Caption:") :].strip() + " "
            )
        elif normalized_line.startswith("Top-Level Element Identification:"):
            mode = "features"  # Stop collecting lines for detailed caption
        elif normalized_line.startswith("Concise Formatted Caption:"):
            mode = "concise"
            # Check if there's text on the same line following 'Concise Caption:'
            inline_concise = normalized_line[
                len("Concise Formatted Caption:") :
            ].strip()
            if inline_concise:
                concise_caption += inline_concise
                break
            else:
                mode = "concise_next_line"
        elif mode == "detailed" and line.strip():
            detailed_caption += line.strip()
        elif mode == "concise_next_line" and line.strip():
            concise_caption += line.strip()
            break  # Assuming concise caption is only one line
        elif mode == "features" and line.startswith("- "):  # Prominent features
            # Use a regular expression to extract the text and
            # the label inside the square brackets
            pattern = r"\[([^\]]+)\]\s*\[([^\]]+)\]"
            text_labels = re.findall(pattern, line)
            if len(text_labels) == 1 and len(processed_texts) < max_n_features:
                feature, label = text_labels[0]
                entity_info = _transform_text_label_to_entity_info(
                    feature, label, "", processed_texts
                )
                if entity_info is not None:
                    identified_entities.append(entity_info)

    if concise_caption:
        query_result = concise_caption_to_query_result(
            concise_caption, node_id, processed_texts
        )
    else:
        query_result = QueryResult(descs=[], entities=[])
    if detailed_caption:
        query_result.descs.insert(
            0, (Description(text=detailed_caption.strip(), label="detail"), [node_id])
        )
    entities = [entity for entity, _ in query_result.entities] + identified_entities
    entities_update = []
    descs = [desc.text for desc, _ in query_result.descs]
    for entity in entities:
        ref_poss = find_ref_poss(descs, entity.text)
        if len(ref_poss) > 0:
            entities_update.append((entity, ref_poss))
    query_result.entities = entities_update
    query_result.raw = text
    return query_result


def concise_caption_to_query_result(
    concise_caption: str, parent_node_id: str = "", processed_texts: Set[str] = None
) -> QueryResult:
    pattern = r"\[([^\]]+)\]\s*\[([^\]]+)\]"
    modified_caption = re.sub(
        pattern, r"\1", concise_caption
    )  # Replace patterns with just the text

    desc = Description(text=modified_caption, label="short")
    descs = [(desc, [parent_node_id])]
    entities = []

    if processed_texts is None:
        processed_texts = set()

    for text, label in re.findall(pattern, concise_caption):
        entity_info = _transform_text_label_to_entity_info(
            text, label, parent_node_id, processed_texts
        )
        # Only add the entity if it's not already in the set
        if entity_info is not None:
            entities.append((entity_info, []))

    return QueryResult(descs=descs, entities=entities)


def _transform_text_label_to_entity_info(
    text: str, label: str, node_id: str, processed_texts: Set[str]
) -> Optional[EntityInfo]:
    text = text.strip().lower()

    # When there are multiple elements, the model has tendency to label them
    # with numbers. We need to remove these numbers to get the actual text.
    if " " in text:
        text_components = text.split(" ")
        if text_components[-1].isdigit():
            text = " ".join(text_components[:-1])
            label = "multiple"
        if len(text_components) > 2 and " ".join(text_components[-2:]) in [
            "on left",
            "on right",
        ]:
            text = " ".join(text_components[:-2])
            label = "multiple"

    if text in processed_texts or text.strip() == "":
        return None
    processed_texts.add(text)

    entity_label = label if label in ["single", "multiple"] else "entity"
    if node_id:
        entity_id = f"{node_id}_{text}"
    else:
        entity_id = text
    entity_info = EntityInfo(text=text, entity_id=entity_id, label=entity_label)
    return entity_info


"""
Parse reply for entity query
"""


def parse_entity(text, action_input: ActionInput) -> QueryResult:
    query_result = QueryResult(descs=[], entities=[], raw=text)
    parse_result_prelim = parse_entity_text(text)
    if parse_result_prelim is None:
        return query_result
    detailed_caption, short_captions, prominent_features = parse_result_prelim
    node_id = action_input.first_entity_id
    desc_texts = []
    if detailed_caption:
        query_result.descs.append(
            (Description(text=detailed_caption, label="detail"), [node_id])
        )
        desc_texts.append(detailed_caption)
    if short_captions:
        for short_caption in short_captions:
            query_result.descs.append(
                (Description(text=short_caption, label="short"), [node_id])
            )
            desc_texts.append(short_caption)
    parent_text = action_input.entity_info.text
    if prominent_features:
        for entity_text, label in prominent_features:
            # Do not add the current entity as new entity
            if plural_to_singular(entity_text) == plural_to_singular(parent_text):
                continue
            entity_label = label if label in ["single", "multiple"] else "entity"
            entity_id = f"{node_id}_{entity_text}"
            entity = EntityInfo(
                text=entity_text, entity_id=entity_id, label=entity_label
            )
            ref_poss = find_ref_poss(desc_texts, entity_text)
            if len(ref_poss) > 0:
                query_result.entities.append((entity, ref_poss))
    return query_result


def parse_entity_text(
    text: str, max_n_features=10
) -> Optional[tuple[Optional[str], list[str], list[tuple[str, str]]]]:
    lines = text.split("\n")
    if "Object Present: No" in text or "Object Presence: No" in text:
        return None

    detailed_caption = ""
    short_captions = []
    prominent_features = []
    seen_features = set()

    mode = None
    for line in lines:
        normalized_line = line.lstrip("#").strip()
        if mode == "detailed" and not line.strip():  # End of detailed caption block
            mode = None
        elif normalized_line.startswith("Detailed Caption:"):
            mode = "detailed"
            detailed_caption = line[len("Detailed Caption:") :].strip()
        elif normalized_line.startswith(
            "Prominent Features:"
        ):  # End of detailed caption block
            mode = None
        elif normalized_line.startswith("Short Captions:"):
            mode = "short"
        elif normalized_line.startswith("Identification of Prominent Features:"):
            mode = "features"
        elif line.startswith("- ") and mode == "short":  # Short captions
            short_caption = line[2:].strip()
            if short_caption and short_caption != "N/A":
                short_captions.append(short_caption)
        elif mode == "features" and line.startswith("- "):  # Prominent features
            if ":" not in line:
                # print(f"Skipping line: {line}")
                continue
            feature, rest = line[2:].split(":", 1)
            # Use a regular expression to extract just
            # the label inside the square brackets
            match = re.match(r"\s*\[([^\]]+)\]", rest.strip())
            if match:
                label = match.group(1).lower()  # Extract the label from the match
                feature = feature.strip().lower()
                if feature in seen_features:
                    continue
                seen_features.add(feature)
                prominent_features.append((feature, label))
                # Limit the number of features as the model
                # may return a long list of random stuff
                if len(prominent_features) == max_n_features:
                    break
        elif mode == "detailed":  # Continue capturing the detailed caption
            detailed_caption += " " + line.strip()
    if detailed_caption == "N/A" or detailed_caption == "":
        detailed_caption = None

    return detailed_caption, short_captions, prominent_features


"""
Parse reply for composition query
"""


def parse_composition(text: str, action_input: ActionInputWithBboxes) -> QueryResult:
    comp_desc, descriptions = parse_composition_text(text)
    entities = []
    for entity_info in action_input.entities:
        ref_poss = find_ref_poss([comp_desc], entity_info.text)
        # Make sure that all the entities appear in compositional caption
        if len(ref_poss) == 0:
            if comp_desc == "":
                comp_desc = entity_info.text
            else:
                comp_desc = comp_desc + " " + entity_info.text
            ref_poss = find_ref_poss([comp_desc], entity_info.text)
        entities.append((entity_info, ref_poss))
    node_id = action_input.first_entity_id
    comp_desc = Description(text=comp_desc, label="composition")
    descriptions = [
        (Description(text=description, label="short"), [node_id])
        for description in descriptions
    ]
    # We can also use find_ref_poss() here for other descriptions
    descriptions = [(comp_desc, [node_id])] + descriptions
    query_result = QueryResult(descs=descriptions, entities=entities, raw=text)
    return query_result


def parse_composition_text(raw_text: str) -> tuple[str, list[str]]:
    # Initialize variables to hold the composition and descriptions
    composition = ""
    descriptions = []

    # Split the text into lines for easier processing
    lines = raw_text.split("\n")

    # Flags to keep track of where we are in the text
    reading_composition = False
    reading_descriptions = False

    for line in lines:
        normalized_line = line.lstrip("#").strip()
        # Check if we're starting the composition section
        if normalized_line.startswith("Composition:"):
            reading_composition = True
            composition = normalized_line.replace("Composition:", "").strip()
            continue

        # Check if we're starting the general descriptions section
        if normalized_line.startswith(
            "General descriptions:"
        ) or normalized_line.startswith("General Descriptions:"):
            reading_composition = False
            reading_descriptions = True
            remaining = (
                normalized_line.replace("General descriptions:", "")
                .replace("General Descriptions:", "")
                .strip()
            )
            if remaining:
                descriptions.append(remaining)
            continue

        # If we're in the composition section, keep appending lines to the composition
        if reading_composition:
            composition += " " + line.strip()

        # If we're in the general descriptions section, add each description to the list
        if reading_descriptions and line.startswith("-"):
            # Extract the description text, removing the leading "- "
            description = line[2:].strip()
            descriptions.append(description)

    return composition, descriptions


"""
Parse reply for relation query
"""


def parse_relation(text: str, action_input: ActionInputWithEntities) -> QueryResult:
    # Extract sentences describing relations
    sentences = extract_relation_sentences(text)

    descs_with_entities: list[tuple[Description, list[str]]] = []

    # Process each sentence to create Description instances and match entities
    for sentence in sentences:
        # Find and replace entity references in the sentence
        modified_sentence, entity_ids = replace_entities_with_text(
            sentence, action_input.entities
        )
        # Do not store the description if it does not contain at least 2 entities
        if len(entity_ids) < 2:
            continue
        # Create a Description instance for the relation
        relation_description = Description(text=modified_sentence, label="relation")
        # Append the relation description and associated entity IDs
        descs_with_entities.append((relation_description, entity_ids))

    # We do not need to use entities here either as they are already
    # present in the action_input
    # However, it could be better to include them here for consistency
    query_result = QueryResult(descs=descs_with_entities, entities=[], raw=text)
    return query_result


def extract_relation_sentences(text: str) -> list[str]:
    # Pattern to match sentences starting with "-" or "number."
    pattern = re.compile(r"(?:^|\n)-\s*([^\n]+)|(?:^|\n)\d+\.\s*([^\n]+)")
    matches = pattern.findall(text)

    # Extract sentences, ignoring None matches in groups
    sentences = [match[0] or match[1] for match in matches]
    sentences = list(set(sentences))
    return sentences


def replace_entities_with_text(
    sentence: str, entities: list[EntityInfo]
) -> tuple[str, list[str]]:
    entity_ids = []
    modified_sentence = sentence

    for entity in entities:
        # Create a pattern to match the entity text as a whole word, case-insensitive
        # Optionally include brackets around the entity text
        pattern = r"\b\[?" + re.escape(entity.text) + r"\]?\b"
        pattern = re.compile(pattern, re.IGNORECASE)

        count = len(pattern.findall(modified_sentence))

        if count > 0:
            entity_ids.append(entity.entity_id)
            bracketed_pattern = re.compile(
                r"\[" + re.escape(entity.text) + r"\]", re.IGNORECASE
            )
            modified_sentence = bracketed_pattern.sub(entity.text, modified_sentence)

    return modified_sentence, entity_ids
