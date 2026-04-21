import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from dataset import SurgicalDataset
from loss import dice_coef, weighted_binary_crossentropy, combined_loss

# ---------------------------------
# 1️ Check argument
# ---------------------------------
if len(sys.argv) != 2:
    print("Usage: python predict.py [baseline/weighted/combined]")
    sys.exit()

mode = sys.argv[1]

if mode not in ["baseline", "weighted", "combined"]:
    print("Invalid option. Use 'baseline', 'weighted', or 'combined'")
    sys.exit()

print(f"\nRunning Prediction Mode: {mode.upper()}\n")

# ---------------------------------
# 2️ Select Model
# ---------------------------------
if mode == "baseline":
    model_path = "models/baseline_model.h5"

elif mode == "weighted":
    model_path = "models/weighted_model.h5"

else:  # combined
    model_path = "models/combined_model.h5"

if not os.path.exists(model_path):
    print("Model not found! Train first.")
    sys.exit()

# ---------------------------------
# 3️ Load Model
# ---------------------------------
model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "dice_coef": dice_coef,
        "weighted_binary_crossentropy": weighted_binary_crossentropy,
        "combined_loss": combined_loss
    },
    compile=False
)

# ---------------------------------
# 4️ Load Dataset
# ---------------------------------
image_dir = "dataset/images"
mask_dir = "dataset/masks"
def numerical_sort(filename): 
    return int(filename.split('_')[1].split('.')[0])
image_files = sorted( 
    [f for f in os.listdir(image_dir) if f.endswith("_endo.png")], 
    key=numerical_sort 
)
dataset = SurgicalDataset(image_dir, mask_dir, image_files, batch_size=1)

# ---------------------------------
# 5️ Create Results Folder
# ---------------------------------
output_folder = f"results/{mode}"
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------
# 6️ Predict All Images
# ---------------------------------
for i in range(len(dataset)):

    image, mask = dataset[i]

    prediction = model.predict(image, verbose=0)
    prediction = (prediction > 0.5).astype(np.uint8)

# Convert to display format
    original = (image[0] * 255).astype(np.uint8)
    gt_mask = (mask[0].squeeze() * 255).astype(np.uint8)
    pred_mask = (prediction[0].squeeze() * 255).astype(np.uint8)

# Convert masks to 3-channel for stacking
    gt_mask_3 = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    pred_mask_3 = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

# Combine horizontally
    combined_image = np.hstack((original, gt_mask_3, pred_mask_3))

# Save
    filename = image_files[i].replace("_endo.png", "_comparison.png")
    save_path = os.path.join(output_folder, filename)

    cv2.imwrite(save_path, combined_image)


print(f"\nAll segmented comparison images saved in: {output_folder}")
print("Prediction Complete ")
