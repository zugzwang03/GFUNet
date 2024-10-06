# Training function
import time
import torch
import UNet
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

def train(model, dataloader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0

    start_time = time.time()  # Record the start time

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs = model(images)
        outputs = torch.sigmoid(outputs)  # Use sigmoid for binary segmentation
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    end_time = time.time()  # Record the end time

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Training Loss: {epoch_loss:.4f}, Time Elapsed: {elapsed_time:.2f} seconds")

    return epoch_loss

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Validation Loss: {epoch_loss}")
    return epoch_loss

# Model, loss function, optimizer, and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_class=1).to(device)  # Set model to GPU if available

criterion = nn.BCELoss()  # Binary Cross Entropy for segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)
