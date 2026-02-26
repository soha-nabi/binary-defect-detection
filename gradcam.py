import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- LOAD BINARY MODEL -----
model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load("model_binary.pth", map_location=device))
model.to(device)
model.eval()

target_layer = model.features[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def generate_gradcam(image_path, save_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    pil_image = Image.fromarray(img_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    img_float = img_rgb.astype(np.float32) / 255
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))