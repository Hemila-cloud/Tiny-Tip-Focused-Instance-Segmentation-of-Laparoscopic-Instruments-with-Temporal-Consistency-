import torch
from torch.utils.data import DataLoader
from temporal_dataset import TemporalDataset
from temporal_model import TemporalUNet
from temporal_loss import combined_temporal_loss
import os

print("=== Temporal Training Started ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = TemporalDataset("dataset/images", "dataset/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = TemporalUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    print(f"\nEpoch {epoch+1}/{epochs}")

    for batch_idx, (frames, masks) in enumerate(loader):

        frames = frames.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(frames)

        loss = combined_temporal_loss(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx} Loss: {loss.item():.4f}")

    print("Average Loss:", total_loss / len(loader))

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/temporal_model.pth")

print("\nTraining Completed Successfully")
