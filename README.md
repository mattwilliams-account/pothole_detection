# pothole_detection
CS-481 Machine Learning project 

# Dataset
- https://universe.roboflow.com/gerapothole/pothole-detection-yolov8/dataset/1

# Required packages (pip)
- ultralytics
- torch (Pytorch)

# Instructions to setup project
- Upload downloaded scripts from github or clone repo using git (if manually uploading files: keep files in the same directory)
- In the same directory the scripts are in, create a directory named "datasets" and move the dataset into this directory
- Again in the directory consisting of the scripts, create a directory named "models" 
- follow instructions depending on locally running or using Anvil
  # Running model with Anvil
  - Login to Anvil: " ssh <user_name>@anvil.rcac.purdue.edu "
  - cd to location of directory with scripts
  - Load conda module for python: " module load anaconda "
  - create python env to install dependencies: " python3 -m venv my_venv1 "
  - activate environment: " source my_venv1/bin/activate "
  - update pip: " pip install -- upgrade pip "
  - install ultralytics & Pytorch: " pip install ultralytics torch "
  - request resource allocation (change time depending on needs): " salloc -- partition = gpu -- gres = gpu :1 -- time =00:02:00 "
  - activate environment again: " source my_venv1 / bin / activate "
  - run the script: " python3 YOLO_script.py "
  - Results will be provided in the "runs" directory created after script completetion.
  # Running model locally
  - navigate to directory with scripts and preivously created directories
  - create python environment to install dependencies: " python3 -m venv my_venv1 "
  - activate environment: " source my_venv1/bin/activate "
  - update pip: " pip install -- upgrade pip "
  - install ultralytics & Pytorch: " pip install ultralytics torch "
  - run the script: " python3 YOLO_script.py "
  - Results will be provided in the "runs" directory created after script completetion.
  
# Model used 
- YOLOv8-nano
  
# Example results
- the "runs1" directory in repo provides examples of results generated from running the script.
  
# Possible expansion
- this project could be expanded upon with the use of different size YOLOv8 models that have more capabilities to detect smaller images and allow for a more complete comparison between model performance and dataset instablity. 
