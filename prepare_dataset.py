import os
import shutil

raw_path = "raw_dataset/video"
image_output = "dataset/images"
mask_output = "dataset/masks"

os.makedirs(image_output, exist_ok=True)
os.makedirs(mask_output, exist_ok=True)

for video_folder in os.listdir(raw_path):
    video_path = os.path.join(raw_path, video_folder)

    if os.path.isdir(video_path):
        for file in os.listdir(video_path):

            # Select only original images
            if file.endswith("_endo.png"):
                src = os.path.join(video_path, file)
                dst = os.path.join(image_output, file)
                shutil.copy(src, dst)

            # Select only binary masks
            if file.endswith("_endo_mask.png"):
                src = os.path.join(video_path, file)
                dst = os.path.join(mask_output, file)
                shutil.copy(src, dst)

print("Dataset Prepared Successfully!")
