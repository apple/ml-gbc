python scripts/generation/gbc2i.py \
    --configs configs/generation/gbc2i/sampling_base.yaml \
    --prompt_file prompts/t2i/t2gbc_seed.yaml
python scripts/generation/gbc2i.py \
    --configs configs/generation/gbc2i/sampling_region_gbc_encode_without_context_ipa.yaml \
    --prompt_files prompts/t2i/dog_cat_ref_image.yaml
python scripts/generation/gbc2i.py \
    --configs configs/generation/gbc2i/sampling_gbc_encode_without_context.yaml \
    --prompt_files prompts/t2i/banana_apple_graph_only.yaml prompts/t2i/living_room_graph_only.yaml
