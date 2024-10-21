import numpy as np
import matplotlib.pyplot as plt
import train_val
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import dataloader

leftOrRight = dataloader.leftOrRight
imageHeight, imageWidth = 0, 0

transform = dataloader.transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = train_val.model

# Load the best model
model.load_state_dict(torch.load("best_unet_model.pth"))
model.eval()

image_dir = "/content/drive/MyDrive/CASIA-Iris-Interval"

output_masks = []
for idx in range(1, 100):
    dir_path = os.path.join(image_dir, f"{idx:03d}", "L")
    if os.path.exists(dir_path):
        for img_file in os.listdir(dir_path):
            if img_file.endswith(".jpg"):
                test_image = Image.open(os.path.join(dir_path, img_file)).convert("RGB")
                imageHeight, imageWidth = test_image.size
                test_image_tensor = (
                    transform(test_image).unsqueeze(0).to(device)
                )  # Add batch dimension

                with torch.no_grad():
                    output = model(test_image_tensor)
                    output = torch.sigmoid(output)
                    output_mask = (
                        output.squeeze(0).cpu().numpy()
                    )  # Remove batch dimension
                    output_masks.append(output_mask)

idx = 0

for output_mask in output_masks:
    idx += 1
    threshold = np.mean(output_mask[0])  # Set a threshold value to binarize the output
    output_mask = (output_mask > threshold).astype(np.float32)
    output_image = Image.fromarray((output_mask[0] * 255).astype("uint8"))  # Convert mask to image
    output_image = output_image.resize((imageHeight, imageWidth))
    save_path = os.path.join("/content/drive/MyDrive/Unet", leftOrRight, f"pred_{idx}.png")  # Update the path to save the mask
    output_image.save(save_path)
    print(f"Output mask saved at {save_path}")
