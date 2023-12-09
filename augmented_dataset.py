import numpy as np
import torch
from torchvision import transforms



class AugmentedSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __getitem__(self, idx):
        image = np.load(self.images[idx])
        mask = np.load(self.masks[idx])

        # Convert float64 to uint8 for image and mask
        image = (image * 255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)

        if self.transform:
            # Convert images and masks to PIL images for transformations
            image_pil = transforms.ToPILImage()(image)
            mask_pil = transforms.ToPILImage()(mask)

            # Apply transformations
            image = np.array(self.transform(image_pil))
            mask = np.array(self.transform(mask_pil))

        return image, mask

    def __len__(self):
        return len(self.images)
