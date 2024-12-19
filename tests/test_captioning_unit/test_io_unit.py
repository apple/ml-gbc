from objprint import op

from gbc.captioning import get_action_input_from_img_path
from gbc.captioning.auto_actions import AutoImageQuery
from gbc.captioning.primitives import (
    NodeInfo,
    QueryResult,
    ActionInputPair,
)


img_path = "data/images/wiki/Eiffel_tower_0.jpg"
action_input = get_action_input_from_img_path(img_path)


print("------ Test NodeInfo -------")

print("Define NodeInfo")
query_result = QueryResult(descs=[], entities=[], raw="")
node_info = NodeInfo(
    action_input=action_input,
    query_result=query_result,
)
print(node_info)

print("Convert to dict")
node_info_dict = node_info.model_dump()
op(node_info_dict)

print("Convert from dict")
node_info = NodeInfo.model_validate(node_info_dict)
print(node_info)

print("Load image")
image = node_info.action_input.get_image()
print(node_info)


print()
print("------ Test ActionInputPair -------")

print("Define ActionInputPair")
action_input_pair = ActionInputPair(
    action_class=AutoImageQuery,
    action_input=action_input,
)
print(action_input_pair)

print("Convert to dict")
action_input_pair_dict = action_input_pair.model_dump()
op(action_input_pair_dict)

print("Convert from dict")
action_input_pair = ActionInputPair.model_validate(action_input_pair_dict)
print(action_input_pair)
