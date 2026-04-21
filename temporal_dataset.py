import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class TemporalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, sequence_length=3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sequence_length = sequence_length

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.images) - (self.sequence_length - 1)

    def __getitem__(self, idx):
        frames = []

        for i in range(self.sequence_length):
            img_name = self.images[idx + i]
            img_path = os.path.join(self.image_dir, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # smaller size
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            frames.append(img)

        frames = np.stack(frames)

        last_img = self.images[idx + self.sequence_length - 1]
        mask_name = last_img.replace("_endo.png", "_endo_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (128, 128))
        mask = mask / 255.0
        mask = np.expand_dims(mask, 0)

        return torch.tensor(frames, dtype=torch.float32), \
               torch.tensor(mask, dtype=torch.float32)
