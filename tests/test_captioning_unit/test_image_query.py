from objprint import op

from gbc.utils import setup_gbc_logger
from gbc.captioning import (
    get_action_input_from_img_path,
)
from gbc.captioning.mllm.llava import LlavaImageQuery

setup_gbc_logger()

img_path = "data/images/wiki/Eiffel_tower_0.jpg"
action_input = get_action_input_from_img_path(img_path)

system_file_image = "prompts/captioning/system_image.txt"
query_file_image = "prompts/captioning/query_image.txt"

llava_image_query = LlavaImageQuery(
    query_file=query_file_image,
    system_file=system_file_image,
)

queries, result, image = llava_image_query.query(action_input)

print("Queries to complete:")
op([query.model_dump() for query in queries])

print("----------------------------------------------")
print("Result:")
op(result.model_dump())

print("----------------------------------------------")
print("Queries to complete:")
op([query.model_dump() for query in queries])
