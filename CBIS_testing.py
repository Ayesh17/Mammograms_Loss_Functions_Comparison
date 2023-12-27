import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tqdm

from dataset import SegmentationDataset
# from unet import UNet
from UNet_P import UNet
from evaluation_metrices import Evaluation_metrices
from AUNet_1 import AUNet_R16
from unet_1 import build_unet
from loss_functions import Semantic_loss_functions
import config


# paths
dataset_path = config.CBIS_DATASET_PATH_2
test_path = os.path.join(dataset_path, "test")

test_image_dataset_path = os.path.join(test_path, "images")
test_mask_dataset_path = os.path.join(test_path, "masks")

# print("..................Testing..............")
# Load the mammogram images and masks path for the test set
all_test_image_npy_paths = sorted(Path(test_image_dataset_path).glob("*.npy"))
all_test_mask_npy_paths = sorted(Path(test_mask_dataset_path).glob("*.npy"))

# Create the test dataset
test_dataset = SegmentationDataset(all_test_image_npy_paths, all_test_mask_npy_paths)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


# Testing loop
# Create a new model instance
model = UNet()
# model = AUNet_R16()
# Load the saved model state
model_path = "models/CBIS-DDSM/model_12.pt"
model.load_state_dict(torch.load(model_path))

metrics = Evaluation_metrices



model.eval()
test_loss = 0
test_accuracy = 0
test_iou = 0
test_dice = 0
test_specificity = 0
test_precision = 0
test_recall = 0
with torch.no_grad():
    test_steps = 0
    for images, masks in test_loader:
        images = images.float()

        # convert CBIS to RGB
        binary_image = np.expand_dims(images, axis=-1)
        # Stack the single-channel array to create an RGB image by replicating the channel
        rgb_image = np.concatenate([binary_image, binary_image, binary_image], axis=-1)
        images = rgb_image

        masks = masks.float()
        # to remove different colored masks
        masks = torch.where(masks > 0, torch.ones_like(masks), torch.zeros_like(masks))
        # masks = (masks - masks.min()) / (masks.max() - masks.min())


        # Resize the target tensor to match the shape of the input tensor
        # print("images_testing", images.shape)
        # Resize the target tensor to match the shape of the input tensor
        images_tensor = torch.as_tensor(images, dtype=torch.float32).clone().detach()
        # images = images_tensor.view(4, 3, 256, 256)
        images = images_tensor.permute(0, 3, 1, 2)

        # print("images_testing", images.shape)
        masks = masks.unsqueeze(1)
        # masks = F.interpolate(masks, size=(512, 512), mode='bilinear')

        # Forward pass
        outputs = model(images)


        ## Calculate the loss
        outputs = torch.sigmoid(outputs)

        accuracy, precision, recall, specificity, dice_coefficient, iou = metrics.calculate_metrics(outputs, masks)

        test_accuracy += accuracy
        test_precision += precision
        test_recall += recall
        test_dice += dice_coefficient
        test_iou += iou
        test_specificity += specificity

    test_steps += 1


# Calculate mean test evaluation metrics
test_steps = len(test_dataset)
mean_test_accuracy = (test_accuracy / test_steps) * 100
mean_test_iou = test_iou / test_steps
mean_test_dice = test_dice / test_steps
mean_test_specificity = test_specificity / test_steps
mean_test_precision = test_precision / test_steps
mean_test_recall = test_recall / test_steps

print("Testing accuracy: {:.2f}%, Testing Precision: {:.4f}, Testing Recall: {:.4f}, Testing iou: {:.4f}, Testing dice: {:.4f}, Testing specificity: {:.4f}".format(mean_test_accuracy, mean_test_precision, mean_test_recall, mean_test_iou, mean_test_dice, mean_test_specificity))


# # Visualize and save the output as a PNG
#
# # Plot some sample outputs, images, masks, or any relevant data
# sample_output = outputs.cpu().numpy().squeeze()  # Assuming a single output from the batch
# sample_image = images[0].permute(1, 2, 0)  # Assuming a single image from the batch
# sample_image = sample_image[:,:,0]
# sample_mask = masks[0].cpu().numpy().squeeze()  # Assuming a single mask from the batch
#
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#
# axes[0].imshow(sample_output, cmap='gray')
# axes[0].set_title('Model Output')
#
# axes[1].imshow(sample_image)
# axes[1].set_title('Input Image')
#
# axes[2].imshow(sample_mask, cmap='gray')
# axes[2].set_title('Ground Truth Mask')
#
# plt.tight_layout()
#
# # Save the plot as a PNG file
# # plt.savefig('output.png')
#
# # Show the plot if needed
# plt.show()


# sample_output = outputs.cpu().numpy().squeeze()  # Assuming a single output from the batch
# plt.imshow(sample_output, cmap='gray')
# plt.show()
