# Binary Defect Detection with Explainability
AI Engineer Intern Assessment — NEU Surface Defect Database

## Overview
This project implements a binary surface defect detection system using deep learning and explainable AI techniques.  
The model classifies steel surface images as **defect vs non-defect** and generates visual explanations using Grad-CAM.

The pipeline includes data preprocessing, transfer learning, inference, and explainability, and is designed to run on CPU.

---

## Dataset
**Dataset used:** NEU Surface Defect Database

The original dataset contains six defect classes.  
For this assessment, a binary classification task was constructed as follows:

- **Defect (1):** Scratches  
- **Non-Defect (0):** All other defect types  

No external data was used.

---

## Model
**Backbone:** EfficientNet-B0 (Transfer Learning)

- Pretrained on ImageNet
- Final classification layer modified for binary output
- Training performed for 5 epochs due to CPU constraints

---

## Explainability
Grad-CAM is used to visualize regions influencing the model’s predictions.

The generated heatmaps highlight scratch regions on defective surfaces, providing interpretability and confirming that the model focuses on relevant defect areas.

---

## Installation

Install dependencies:
pip install -r requirements.txt

---

## Training (Optional)

To retrain the model:
python train.py


This will produce:

- `model_binary.pth`

---

## Inference + Grad-CAM

Place test images inside:


test_images/

Run:
python infer.py


Outputs will be saved in:
```
outputs/
├── predictions.csv
└── Grad-CAM visualization images
```
---

## Output

The inference pipeline:

- Classifies each image as defect or non-defect
- Saves predictions to a CSV file
- Generates Grad-CAM overlays for explainability

---

## Deployment Notes

- Runs on CPU without requiring a GPU
- Suitable for batch inspection scenarios
- Lightweight and reproducible

---

## Project Structure

project/
├── train.py
├── infer.py
├── gradcam.py
├── model_binary.pth
├── requirements.txt
├── report.pdf
├── test_images/
└── outputs/

## Summary

This project demonstrates an end-to-end explainable defect detection pipeline using transfer learning and Grad-CAM.  
The system provides accurate predictions along with interpretable visual evidence, making it suitable for industrial quality inspection applications.
