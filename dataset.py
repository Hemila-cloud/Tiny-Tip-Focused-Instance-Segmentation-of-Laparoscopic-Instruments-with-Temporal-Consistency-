import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

IMG_SIZE = 256

class SurgicalDataset(Sequence):
    def __init__(self, image_dir, mask_dir, file_list, batch_size=8):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.images = file_list   # <-- important

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.images[idx*self.batch_size:(idx+1)*self.batch_size]

        imgs = []
        masks = []

        for img_name in batch_images:

            img_path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0


            mask_name = img_name.replace("_endo.png", "_endo_mask.png")
            mask_path = os.path.join(self.mask_dir, mask_name)

            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            # Convert to binary: instrument = 1, background = 0
            mask = (mask > 0).astype(np.float32)

            mask = np.expand_dims(mask, axis=-1)


            imgs.append(img)
            masks.append(mask)

        return np.array(imgs), np.array(masks)
