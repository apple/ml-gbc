from objprint import op
from omegaconf import OmegaConf

from gbc.utils import setup_gbc_logger
from gbc.captioning import (
    AutoImageQuery,
    AutoEntityQuery,
    get_action_input_from_img_path,
)


img_path = "data/images/wiki/Eiffel_tower_0.jpg"
action_input = get_action_input_from_img_path(img_path)
config = OmegaConf.load("configs/captioning/default.yaml")

setup_gbc_logger()


###

print("Testing AutoImageQuery with config...")
image_query = AutoImageQuery(config)

print("Testing caching")
image_query = AutoImageQuery(config)
queries, result, image = image_query.query(action_input)

print("Queries to complete:")
op([query.model_dump() for query in queries])

print("Result:")
op(result.model_dump())


###

print("----------------------------------------------")

print("Testing AutoEntityQuery with config...")
entity_query = AutoEntityQuery(config)

print("Testing caching")
entity_query = AutoEntityQuery(config)
action_input.entity_info.text = "tower"
queries, result, image = entity_query.query(action_input)

print("Queries to complete:")
op([query.model_dump() for query in queries])

print("Result:")
op(result.model_dump())


###

print("----------------------------------------------")

print("Testing AutoImageQuery...")

image_query = AutoImageQuery()

print("Testing AutoEntityQuery...")

entity_query = AutoEntityQuery()
