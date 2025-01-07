from omegaconf import OmegaConf
from objprint import op

from gbc.utils import save_list_to_file, setup_gbc_logger
from gbc.captioning import GbcPipeline

setup_gbc_logger()

config = OmegaConf.load("configs/captioning/default.yaml")
gbc_pipeline = GbcPipeline.from_config(config)

img_file_1 = "data/images/wiki/Eiffel_tower_0.jpg"
img_file_2 = "data/images/wiki/Eiffel_tower_1.jpg"

# Perform captioning on a single image
gbc = gbc_pipeline.run_gbc_captioning(img_file_1)
# Pretty print the GBC graph
op(gbc[0].model_dump())

# Perform captioning on multiple images
gbcs = gbc_pipeline.run_gbc_captioning([img_file_1, img_file_2])
# Save the GBC graphs, can save as json, jsonl, or parquet
save_list_to_file(gbcs, "tests/outputs/captioning/gbc_eiffel_tower.json")
