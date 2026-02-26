import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# -------- DEVICE --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- TRANSFORMS --------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor()
])

# -------- CUSTOM BINARY DATASET --------
class BinaryNEUDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.paths = []
        self.labels = []
        self.transform = transform

        for defect_type in os.listdir(root_dir):
            defect_path = os.path.join(root_dir, defect_type)

            for img in os.listdir(defect_path):
                self.paths.append(os.path.join(defect_path, img))

                # ⭐ Scratches = defect (1)
                if defect_type == "scratches":
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# -------- DATASET --------
train_dataset = BinaryNEUDataset(
    "dataset/NEU-DET/train/images",
    train_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -------- MODEL --------
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, 2)
model = model.to(device)

# -------- LOSS & OPT --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------- TRAIN LOOP --------
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {running_loss:.4f}")

# -------- SAVE MODEL --------
torch.save(model.state_dict(), "model_binary.pth")
print("Binary model saved as model_binary.pth ✅")