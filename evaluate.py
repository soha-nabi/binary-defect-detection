import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import os

# ---- DEVICE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- LOAD MODEL ----
model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load("model_binary.pth", map_location=device))
model.to(device)
model.eval()

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---- DATASET (Validation Folder) ----
val_dir = "dataset/NEU-DET/validation/images"

y_true = []
y_pred = []

for defect_type in os.listdir(val_dir):
    folder = os.path.join(val_dir, defect_type)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        image = Image.open(img_path)
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()

        # Binary mapping (same as training)
        true_label = 1 if defect_type == "scratches" else 0

        y_true.append(true_label)
        y_pred.append(pred)

# ---- METRICS ----
print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))