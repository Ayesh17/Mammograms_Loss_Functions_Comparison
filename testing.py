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
dataset_path = config.DATASET_PATH
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

train_image_dataset_path = os.path.join(train_path, "images")
train_mask_dataset_path = os.path.join(train_path, "masks")

test_image_dataset_path = os.path.join(train_path, "images")
test_mask_dataset_path = os.path.join(train_path, "masks")

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
# Load the saved model state
model_path = "models/model_1.pt"
model.load_state_dict(torch.load(model_path))

metrics = Evaluation_metrices

# Define the loss function
# loss_function = nn.BCELoss()
# loss_function = nn.CrossEntropyLoss()
loss_function = Semantic_loss_functions()


model.eval()
test_loss = 0
test_accuracy = 0
test_iou = 0
test_dice = 0
test_specificity = 0
test_recall = 0
with torch.no_grad():
    for images, masks in test_loader:
        images = images.float()

                # # convert CBIS to RGB
                # binary_image = np.expand_dims(images, axis=-1)
                # # Stack the single-channel array to create an RGB image by replicating the channel
                # rgb_image = np.concatenate([binary_image, binary_image, binary_image], axis=-1)
                # images = rgb_image

        masks = masks.float()
        masks = (masks - masks.min()) / (masks.max() - masks.min())


        # Resize the target tensor to match the shape of the input tensor
        # print("images_testing", images.shape)
        # Resize the target tensor to match the shape of the input tensor
        images_tensor = torch.as_tensor(images, dtype=torch.float32).clone().detach()
        images = images_tensor.view(4, 3, 256, 256)

        # print("images_testing", images.shape)
        masks = masks.unsqueeze(1)
        # masks = F.interpolate(masks, size=(512, 512), mode='bilinear')

        # Forward pass
        outputs = model(images)


        ## Calculate the loss
        outputs = torch.sigmoid(outputs)
        # loss = loss_function(outputs, masks)
        # loss = loss_function.dice_loss(outputs, masks)
        loss = loss_function.bce_dice_loss(outputs, masks)
        test_loss += loss

        tp, tn, fp, fn = metrics.calculate_metrics(outputs, masks)

        # calculate accuracy
        accuracy = metrics.calclate_accuracy(tp, tn, fp, fn)
        # print("accuracy", accuracy)
        test_accuracy += accuracy

        # calculate accuracy
        recall = metrics.calculate_recall(tp, tn, fp, fn)
        # print("recall", recall)
        test_recall += recall

        # calculate accuracy
        dice_coefficient = metrics.calculate_dice_coefficient(tp, tn, fp, fn)
        # print("dice_coefficient", dice_coefficient)
        test_dice += dice_coefficient

        # calculate the iou
        iou = metrics.calculate_iou(tp, tn, fp, fn)
        test_iou += iou
        # print(f"testing IoU: {test_iou}")

        # calculate the iou
        specificity = metrics.calculate_specificity(tp, tn, fp, fn)
        test_specificity += specificity
        # print(f"testing specificity: {test_specificity}")


# Calculate mean test evaluation metrics
test_steps = len(test_dataset)
mean_test_loss = test_loss / test_steps
mean_test_accuracy = (test_accuracy / test_steps) * 100
mean_test_iou = test_iou / test_steps
mean_test_dice = test_dice / test_steps
mean_test_specificity = test_specificity / test_steps
mean_test_recall = test_recall / test_steps

print("Testing accuracy: {:.2f}%, Testing Loss: {:.4f}, Testing Sensitivity: {:.4f}, Testing iou: {:.4f}, Testing dice: {:.4f}, Testing specificity: {:.4f}".format(mean_test_accuracy, mean_test_loss, mean_test_recall, mean_test_iou, mean_test_dice, mean_test_specificity))

