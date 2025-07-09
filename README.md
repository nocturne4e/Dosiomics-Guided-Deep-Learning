# Dosiomics-Guided Deep Learning for Radiation Esophagitis Prediction in Lung Cancer
An architecture using ResNet as the backbone was developed to fuse DL features with dosiomic features while incorporating prior knowledge to guide the prediction model.
# Data Collection
Using patient CT images and radiation dose distribution images from three hospitals.

# Methodology
DenseNet-121
ResNet-34
ResNet-50
MobileNet
Contrastive Learning
Radiomics

# Usage
Prepare Environment
Recommended using virtual environment
python3 -m venv .env
Running on Python3.8
pip install -U -r requirements.txt

# Training
python main.py

# Performance Evaluation
AUC
Specitivity
Sensitivity
