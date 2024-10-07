import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

leftOrRight = 'R'

# Sample custom dataset for image segmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Collect images from multiple directories
        self.images = []
        for idx in range(1, 250):
            dir_path = os.path.join(image_dir, f"{idx:03d}", leftOrRight)
            if os.path.exists(dir_path):
                for img_file in os.listdir(dir_path):
                    if img_file.endswith('.jpg'):
                        self.images.append(os.path.join(dir_path, img_file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_filename = os.path.basename(img_path)
        mask_filename = 'OperatorA_' + img_filename.replace(".jpg", ".tiff")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale for mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask to float32
        mask = mask.float()

        return image, mask

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

image_dir = "/content/drive/MyDrive/CASIA-Iris-Interval"
mask_dir = "/content/drive/MyDrive/casia4i"

# Create datasets and dataloaders
train_dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

val_dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
