import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class TIFDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (str): Path to the directory containing .tif images.
            transform (callable, optional): Transform to be applied to each image.
        """
        self.directory = directory
        self.transform = transform
        self.image_paths = [
            os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency
        if self.transform:
            image = self.transform(image)
        return image

# Directory containing .tif images
image_directory = "/path/to/your/tif/images"

# Define transformations with random cropping
transform = transforms.Compose([
    transforms.RandomCrop((256, 256)),  # Randomly crop to 256x256
    transforms.ToTensor(),              # Convert to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Create dataset and dataloader
dataset = TIFDataset(directory=image_directory, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Example usage
for batch in dataloader:
    print(batch.shape)  # Outputs: torch.Size([16, 3, 256, 256])
