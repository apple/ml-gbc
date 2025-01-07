import cv2
from objprint import op

from gbc.utils import setup_gbc_logger
from gbc.captioning.detection.yolo_world import YoloWorldDetection


setup_gbc_logger()

print("Testing YoloWorldDetection X...")

detection_model = YoloWorldDetection(model_version="x_v2")

img_path = "data/images/wiki/Eiffel_tower_0.jpg"
texts = ["boats", "tower", "train"]
image = cv2.imread(img_path)

print("Running detection...")
bboxes = detection_model.detect(image, texts)

print("----------------------------------------------")
for text, bboxes_per_text in bboxes.items():
    print(f"Detected bounding boxes for '{text}':")
    op([bbox.model_dump() for bbox in bboxes_per_text])


print("----------------------------------------------")

print("Testing YoloWorldDetection L...")

detection_model = YoloWorldDetection(model_version="l_v2")

print("Running detection...")
bboxes = detection_model.detect(image, texts)

print("----------------------------------------------")
for text, bboxes_per_text in bboxes.items():
    print(f"Detected bounding boxes for '{text}':")
    op([bbox.model_dump() for bbox in bboxes_per_text])
