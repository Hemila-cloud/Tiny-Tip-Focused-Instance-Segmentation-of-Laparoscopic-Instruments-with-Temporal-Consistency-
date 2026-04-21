from dataset import SurgicalDataset
import matplotlib.pyplot as plt

dataset = SurgicalDataset("dataset/images", "dataset/masks", batch_size=2)

images, masks = dataset[0]

print("Image shape:", images.shape)
print("Mask shape:", masks.shape)

plt.subplot(1,2,1)
plt.imshow(images[0])
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(masks[0].squeeze(), cmap="gray")
plt.title("Mask")

plt.show()
