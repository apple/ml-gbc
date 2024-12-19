CMAKE_ARGS="-DLLAMA_CUDA=on" python -m pip install llama-cpp-python==0.2.79
python -m pip install huggingface_hub
python scripts/setup/download_llava_models.py
