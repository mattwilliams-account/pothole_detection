import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# load env file
load_dotenv()

# project directory
#project_name = os.path.basename(os.getcwd())
#scratch_dir = os.path.join(os.environ.get("SCRATCH", "/anvil/scratch/x-ajana1"), project_name)
scratch_dir = os.path.join("C:\Users\willi\OneDrive\Documents\School\Spring 2026\CS-481 ML\ml_project\CS-481-project\pothole_detection")

# paths
dataset_dir = os.path.join(scratch_dir, "datasets", "300-300_road_img")
model_dir = os.path.join(scratch_dir, "models")
project_dir = os.getcwd()
# hf_token_file = os.path.join(project_dir, "hf_tok.txt")
hf_token = os.getenv("hfTok")
# make directories
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# read hf token
# if os.path.exists(hf_token_file):
#     with open(hf_token_file, "r") as f:
#         hf_token = f.read().strip()
#     print("hugging face token loaded")
# else:
#     print("hugging face token file not found")
#     exit(1)

# model name and path
model_name = "llava-hf/llava-1.5-7b-hf"
model_save_path = os.path.join(model_dir, "llava-7b")

# download and save model
if not os.path.exists(model_save_path):
    print("downloading llava 7b model from hugging face")
    snapshot_download(
    repo_id="llava-hf/llava-1.5-7b-hf",
    token=hf_token,
    local_dir=model_save_path,
    local_dir_use_symlinks=False  # Optional: ensures full copy
)
    print("llava 7b model and processor saved")
else:
    print("llava 7b model already exists")

# the coco 2017 dataset was downloaded and organized separately using  copy_coco.py script
