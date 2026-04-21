import torch
from torch.utils.data import DataLoader
from temporal_dataset import TemporalDataset
from temporal_model import TemporalUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TemporalDataset("dataset/images", "dataset/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=False)

model = TemporalUNet().to(device)
model.load_state_dict(torch.load("models/temporal_model.pth", map_location=device))
model.eval()

dice_total = 0

with torch.no_grad():
    for frames, masks in loader:

        frames = frames.to(device)
        masks = masks.to(device)

        outputs = model(frames)
        outputs = torch.sigmoid(outputs)
        preds = (outputs > 0.2).float()

        intersection = (preds * masks).sum()
        dice = (2 * intersection + 1e-6) / (preds.sum() + masks.sum() + 1e-6)

        dice_total += dice.item()

print("Average Dice:", dice_total / len(loader))
