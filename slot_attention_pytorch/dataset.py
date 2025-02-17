import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class CustomDataset(Dataset):
    def __init__(self, data_path="data.npy", transform=None):
        super(CustomDataset, self).__init__()
        self.data = np.load(data_path)  # Shape: (12000, 64, 64, 3)
        self.transform = transform

    def __getitem__(self, index):
        image = self.data[index]  # Get image at index
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to (C, H, W)

        # Resize to (128, 128)
        image = TF.resize(image, [32, 32])

        if self.transform:
            image = self.transform(image)

        return {'image': image}

    def __len__(self):
        return len(self.data)