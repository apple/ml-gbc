import cv2
from objprint import op

from gbc.utils import setup_gbc_logger
from gbc.captioning.detection.grounding_dino import GroundingDinoDetection


setup_gbc_logger()

print("Testing GroundingDinoDetection...")

detection_model = GroundingDinoDetection()

img_path = "data/images/wiki/Eiffel_tower_0.jpg"
texts = ["boats", "tower", "train"]
image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

print("Running detection...")
# UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.  # noqa
bboxes = detection_model.detect(image, texts)

print("----------------------------------------------")
for text, bboxes_per_text in bboxes.items():
    print(f"Detected bounding boxes for '{text}':")
    op([bbox.model_dump() for bbox in bboxes_per_text])
