from objprint import op
from omegaconf import OmegaConf

import cv2

from gbc.utils import setup_gbc_logger
from gbc.data.bbox.annotate import annotate_all_labels
from gbc.captioning import (
    AutoDetectionActionFromImage,
    AutoDetectionActionFromEntity,
)
from gbc.captioning.primitives import (
    EntityInfo,
    ActionInputWithEntities,
    get_action_input_from_img_path,
)
from gbc.captioning.detection.detection_action import DetectionAction


###

setup_gbc_logger()

img_path = "data/images/wiki/Eiffel_tower_0.jpg"
action_input = get_action_input_from_img_path(img_path)

entities = [
    EntityInfo(label="multiple", text="boats", entity_id="boats"),
    EntityInfo(label="single", text="tower", entity_id="tower"),
    EntityInfo(label="single", text="train", entity_id="train"),
]

action_input_with_entities = ActionInputWithEntities(
    image=action_input.image,
    entity_info=action_input.entity_info,
    entities=entities,
)

config = OmegaConf.load("configs/captioning/default.yaml")


###

# The first time detection model is loaded
print("Testing DetectionAction...")

detection_action = DetectionAction()
queries, _, _ = detection_action.query(action_input_with_entities)

op([query.model_dump() for query in queries])


###

print("-------------------------------------------------------------")
print("Testing AutoDetectionActionFromImage with config...")

auto_detection_action = AutoDetectionActionFromImage(config)

print("-------------------------------------------------------------")
print("Testing caching...")
auto_detection_action = AutoDetectionActionFromImage(config)
queries, _, _ = auto_detection_action.query(action_input_with_entities)

op([query.model_dump() for query in queries])

labeled_bboxes = []
for query in queries:
    labeled_bboxes.append(
        (
            query.action_input.first_entity_id,
            query.action_input.bbox,
        )
    )
# Note that cv2 is in BGR instead of RGB
img_annotated = annotate_all_labels(
    action_input.get_image(return_pil=False)[:, :, ::-1], labeled_bboxes
)
save_img_path = "tests/outputs/detection/annotated_eiffel_tower.jpg"
cv2.imwrite(save_img_path, img_annotated)


###

print("-------------------------------------------------------------")
print("Testing AutoDetectionActionFromImage without config...")
auto_detection_action = AutoDetectionActionFromImage()
queries, _, _ = auto_detection_action.query(action_input_with_entities)

op([query.model_dump() for query in queries])


###

print("-------------------------------------------------------------")
print("Testing AutoDetectionActionFromEntity with config...")

auto_detection_action = AutoDetectionActionFromEntity(config)
queries, _, _ = auto_detection_action.query(action_input_with_entities)

op([query.model_dump() for query in queries])
