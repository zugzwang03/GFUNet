import train_val 
import model
import dataloader
import torch
import time

train = train_val.train
validate = train_val.validate
model = model.model
train_loader = dataloader.train_loader
val_loader = dataloader.val_loader
criterion = train_val.criterion
optimizer = train_val.optimizer
device = train_val.device

# Number of epochs to train
n_epochs = 5
best_loss = float('inf')

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    start_time = time.time()
    # Training
    train_loss = train(model, train_loader, criterion, optimizer, device)

    # Validation
    val_loss = validate(model, val_loader, criterion, device)
    print('Time is:')
    print(time.time() - start_time)
    # Save the model with the best validation loss
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_unet_model.pth")
        print(f"Model saved at epoch {epoch+1}")

print("Training Completed.")
