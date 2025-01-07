from PIL import Image

from gbc.utils import setup_gbc_logger
from gbc.captioning.mllm.pixtral import load_pixtral_model, pixtral_query_single


setup_gbc_logger()


with open("prompts/captioning/query_image.txt", "r") as f:
    query = f.read()

with open("prompts/captioning/system_image.txt", "r") as f:
    system_message = f.read()


image = Image.open("data/images/wiki/Eiffel_tower_0.jpg").convert("RGB")


print("Testing pixtral_query_single with image query...")

model = load_pixtral_model("nm-testing/pixtral-12b-FP8-dynamic")
result = pixtral_query_single(model, image, query, system_message=system_message)
print(result)
