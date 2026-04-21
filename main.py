import os
from PIL import Image # Pillow is the package name
from sklearn.model_selection import train_test_split # scikit-learn
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor


# need to load model (transformers?)
SCRATCH_DIR = os.path.join("C:\Users\willi\OneDrive\Documents\School\Spring 2026\CS-481 ML\ml_project\CS-481-project\pothole_detection") 
MODEL_PATH = os.path.join(SCRATCH_DIR, "models", "llava-7b")
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

print("loading model and processor from:", MODEL_PATH)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
processor = AutoProcessor.from_pretrained(MODEL_PATH)



# load dataset
N_DATA_DIR = os.path.join(SCRATCH_DIR, "datasets", "300-300_road_img", "normal")
P_DATA_DIR = os.path.join(SCRATCH_DIR, "datasets", "300-300_road_img", "potholes")
N_DATA = []
P_Data = []
for file in os.listdir(N_DATA_DIR):
    N_DATA.append({ "img_path": file,
                    "label": "normal"
                   })
for file in os.listdir(P_DATA_DIR):
    P_Data.append({
        "img_path": file,
        "label": "pothole" 
        })

# merge datasets and make train test splits (sklearn train test split)
X_train, X_test, y_train, y_test = train_test_split(N_DATA, P_Data, 0.3, 0.7, shuffle=True, random_state=42)
print(
    "X_train: ", X_train.shape(), "\n"
    "y_train: ", y_train.shape(), "\n"
    "X_test", X_test.shape(), "\n",
    "y_test: ", y_test.shape(), "\n"
)
# samples = []
# for sample in X_train:
#     samples.append(sample)

# responses = []
# for sample in samples:
# Show a sample
sample = N_DATA[0]
print("label:", sample["label"])
print("img_path:", sample["img_path"])
image = Image.open(sample["img_path"]).convert("RGB")
question = "<image>\nTrue or False: there is a pothole present in this road"

# process inputs
inputs = processor(text=question, images=image, return_tensors="pt").to(device)

# generate output
print("generating label...")
output = model.generate(**inputs, max_new_tokens=50)
response = processor.decode(output[0], skip_special_tokens=True)
# responses.append({
#     "response":response,
#     "img_path": sample["img_path"],
#     "label": sample["label"]
#         })
# print output
print("question:", question)
print("model response:", response)
