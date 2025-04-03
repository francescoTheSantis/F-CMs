import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define paths and parameters
data_dir = "./data"
batch_size = 64
image_size = 128  # Resize images to 128x128
validation_split = 0.2

# Define transformations
transform = transforms.Compose([
    transforms.CenterCrop(178),  # Crop to 178x178 to remove image borders
    transforms.Resize((image_size, image_size)),  # Resize to image_size
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Download and load the dataset
dataset = datasets.CelebA(root=data_dir, split="train", download=True, transform=transform)

# Split into training and validation sets
val_size = int(len(dataset) * validation_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Example usage
for images, labels in train_loader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")
    break
