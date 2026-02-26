import os
import csv
import torch
from torchvision import models, transforms
from PIL import Image
from gradcam import generate_gradcam
print("Script started")
print("Files in test_images:", os.listdir("test_images") if os.path.exists("test_images") else "Folder not found")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- LOAD MODEL -----
model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load("model_binary.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

input_folder = "test_images"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

results = []

for file in os.listdir(input_folder):
    if not file.endswith(".jpg"):
        continue

    img_path = os.path.join(input_folder, file)
    image = Image.open(img_path)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    label = "defect" if pred == 1 else "non_defect"

    heatmap_path = os.path.join(output_folder, file)
    generate_gradcam(img_path, heatmap_path)

    results.append([file, label])

# ----- SAVE CSV -----
with open(os.path.join(output_folder, "predictions.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "prediction"])
    writer.writerows(results)

print("Inference completed 🚀")