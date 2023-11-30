import numpy as np
import torch

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __getitem__(self, idx):
        image = np.load(self.images[idx])
        mask = np.load(self.masks[idx])

        return image, mask

    def __len__(self):
        return len(self.images)