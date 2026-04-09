import os
from PIL import Image


# need to load model (transformers?) 


# load dataset
N_DATA_DIR = os.path.join("C:\Users\willi\OneDrive\Documents\School\Spring 2026\CS-481 ML\project\datasets\300-300_road_img\normal")
P_DATA_DIR = os.path.join("C:\Users\willi\OneDrive\Documents\School\Spring 2026\CS-481 ML\project\datasets\300-300_road_img\potholes")
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

# need to merge datasets in random order and make train test splits (sklearn train test split)


samples = []

# Show a sample
sample = samples[0]
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

# print output
print("question:", question)
print("model response:", response)
