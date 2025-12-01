from huggingface_hub import login, hf_hub_download
import os

# Login to HuggingFace
login()

# Download UNI weights
local_dir = "./uni_weights/"
os.makedirs(local_dir, exist_ok=True)

hf_hub_download(
    "MahmoodLab/UNI",
    filename="pytorch_model.bin",
    local_dir=local_dir
)

print("UNI weights downloaded to ./uni_weights/")
