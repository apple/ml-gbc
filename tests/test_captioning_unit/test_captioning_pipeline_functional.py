from omegaconf import OmegaConf
from objprint import op

from gbc.utils import save_list_to_file, setup_gbc_logger
from gbc.captioning import run_gbc_captioning

setup_gbc_logger()

config = OmegaConf.load("configs/captioning/default.yaml")

img_file_1 = "data/images/wiki/Eiffel_tower_0.jpg"
img_file_2 = "data/images/wiki/Eiffel_tower_1.jpg"

# Perform captioning on a single image
gbc = run_gbc_captioning(img_file_1, config, include_relation_query=False)
# Pretty print the GBC graph
op(gbc[0].model_dump())

# Perform captioning on multiple images
gbcs = run_gbc_captioning(
    [img_file_1, img_file_2], config, batch_query=True, batch_size=8
)
# Save the GBC graphs, can save as json, jsonl, or parquet
save_list_to_file(gbcs, "tests/outputs/captioning/gbc_batch_eiffel_tower.json")
