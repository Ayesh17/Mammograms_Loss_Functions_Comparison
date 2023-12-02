import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tqdm

from dataset import SegmentationDataset
from unet import UNet
from evaluation_metrices import Evaluation_metrices
from AUNet_1 import AUNet_R16
from unet_1 import build_unet
from loss_functions import Semantic_loss_functions
import config




# Load the mammogram images and masks path
all_image_npy_paths = sorted(Path(config.IMAGE_DATASET_PATH).glob("*.npy"))
all_mask_npy_paths = sorted(Path(config.MASK_DATASET_PATH).glob("*.npy"))


# Splitting the dataset into train and test
train_images, val_images = train_test_split(all_image_npy_paths, test_size=0.2, random_state=42)
train_masks, val_masks = train_test_split(all_mask_npy_paths, test_size=0.2, random_state=42)

# Split the data into train and validation sets
# train_images, val_images = images[:int(0.8 * len(images))], images[int(0.8 * len(images)):]
# train_masks, val_masks = masks[:int(0.8 * len(masks))], masks[int(0.8 * len(masks)):]

# Create the training and validation datasets
train_dataset = SegmentationDataset(train_images, train_masks)
val_dataset = SegmentationDataset(val_images, val_masks)

print(f"[INFO] found {len(train_dataset)} examples in the training set...")
print(f"[INFO] found {len(val_dataset)} examples in the test set...")


# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False)

# calculate steps per epoch for training and test set
train_steps = len(train_dataset) // config.TRAIN_BATCH_SIZE
val_steps = len(val_dataset) // config.TEST_BATCH_SIZE


# Create the model
model = AUNet_R16()
# model = UNet()
# model = build_unet()

# Define the loss function
# loss_function = nn.BCELoss()
# loss_function = nn.CrossEntropyLoss()
loss_function = Semantic_loss_functions()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr = config.Learning_rate)

#evaluation metrices
metrics = Evaluation_metrices

def convert_to_binary(mask):
    """Converts a mask to binary.

    Args:
        mask: A numpy array representing the mask.

    Returns:
        A numpy array representing the binary mask.
    """

    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    binary_mask[mask > 0] = 255
    return binary_mask



# initialize a dictionary to store training history
H = {"train_accuracy": [], "val_accuracy": [], "train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [], "train_dice": [], "val_dice": [], "train_pixel_accuracy": [], "val_pixel_accuracy": [] , "train_recall": [], "val_recall": [] }

# Train the model
for epoch in range(config.EPOCHS):
    print("Epoch :", epoch+1)

    # print(".............Training.............")
    # Train loop
    model.train()
    train_loss = 0
    train_accuracy = 0
    train_iou = 0
    train_dice = 0
    train_pixel_accuracy = 0
    train_recall = 0
    train_loop = enumerate(tqdm.tqdm(train_loader, total=len(train_loader), leave=True))
    for (i, (images, masks)) in train_loop:
        # print("Training batch:", i)
        # print("image ", images )



        images = images.float()
        masks = masks.float()
        masks = (masks - masks.min()) / (masks.max() - masks.min())

        print("images", images.size())
        # Resize the target tensor to match the shape of the input tensor
        images = images.to(torch.float).view(16, 3, 256, 256)
        masks = masks.unsqueeze(1)
        # masks = F.interpolate(masks, size=(512, 512), mode='bilinear')

        # Forward pass
        outputs = model(images)
        outputs = F.interpolate(outputs, size=(256, 256), mode='nearest')

        # Assuming 'outputs' is the tensor representing the predicted segmentation mask
        outputs = torch.sigmoid(outputs)
        # print("outputs",torch.min(outputs), torch.max(outputs))
        # print("masks",torch.min(masks), torch.max(masks))
        # Calculate the loss
        # loss = loss_function(outputs, masks)
        # print("Mean Loss:", loss.item())
        loss = loss_function.focal_loss(outputs, masks)
        train_loss += loss


        # Calculate the accuracy
        accuracy = (outputs > 0.5).float().mean()
        train_accuracy += accuracy

        # convert the mask to binary
        binary_mask = convert_to_binary(masks)

        # calculate the iou
        iou = metrics.calculate_iou(outputs, binary_mask)
        train_iou += iou
        # print(f"Training IoU: {train_iou}")

        # calculate the dice_score
        dice = metrics.calculate_dice_coefficient(outputs, binary_mask)
        train_dice += dice
        # print(f"Training Dice Coefficient: {train_dice}")

        # calculate the pixel_wise_accuracy
        pixel_accuracy = metrics.calculate_pixel_wise_accuracy(outputs, binary_mask)
        train_pixel_accuracy += pixel_accuracy
        # print(f"Training Pixel-wise Accuracy: {train_pixel_accuracy}")

        # calculate the sensitivity
        recall = metrics.calculate_recall(outputs.cpu().detach().numpy(), masks.cpu().detach().numpy())
        train_recall += recall

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the accuracy
        # print(f'Accuracy: {accuracy}')

    # Print the accuracy
    train_accuracy = (train_accuracy / train_steps) * 100
    train_loss = train_loss / train_steps
    train_iou = train_iou / train_steps
    train_dice = train_dice / train_steps
    train_pixel_accuracy = train_pixel_accuracy / train_steps
    train_recall = train_recall / train_steps


    # print("..................Validation..............")
    # Validation loop
    model.eval()
    val_loss = 0
    val_accuracy = 0
    val_iou = 0
    val_dice = 0
    val_pixel_accuracy = 0
    val_recall = 0
    with torch.no_grad():
        val_loop = enumerate(tqdm.tqdm(val_loader, total=len(val_loader), leave=True))
        for (i, (images, masks)) in val_loop:
            # print("Validation batch:", i)

            images = images.float()
            masks = masks.float()
            masks = (masks - masks.min()) / (masks.max() - masks.min())


            # Resize the target tensor to match the shape of the input tensor
            images = images.to(torch.float).view(8, 1, 256, 256)
            masks = masks.unsqueeze(1)
            # masks = F.interpolate(masks, size=(512, 512), mode='bilinear')

            # Forward pass
            outputs = model(images)
            # are_elements_between_zero_and_one = (outputs >= 0) & (outputs <= 1)
            # result = torch.all(are_elements_between_zero_and_one)
            # print("outputs1", result)
            # print(outputs)

            outputs = F.interpolate(outputs, size=(256, 256), mode='nearest')
            #
            # # Calculate the loss
            # print("outputs", outputs.shape)
            # print("masks", masks.shape)
            # are_elements_between_zero_and_one = (outputs >= 0) & (outputs <= 1)
            # result = torch.all(are_elements_between_zero_and_one)
            # print("outputs2",result)
            #
            # are_elements_between_zero_and_one = (masks >= 0) & (masks <= 1)
            # result = torch.all(are_elements_between_zero_and_one)
            # print("masks",result)

            outputs = torch.sigmoid(outputs)
            # loss = loss_function(outputs, masks)
            loss = loss_function.focal_loss(outputs, masks)
            val_loss += loss

            # Calculate the accuracy
            accuracy = (outputs > 0.5).float().mean()
            val_accuracy += accuracy

            # convert the mask to binary
            binary_mask = convert_to_binary(masks)

            # calculate the iou
            iou = metrics.calculate_iou(outputs, binary_mask) * 100
            val_iou += iou
            # print(f"Validation IoU: {val_iou}")

            # calculate the iou
            dice = metrics.calculate_dice_coefficient(outputs, binary_mask)
            val_dice += dice
            # print(f"Validation Dice Coefficient: {val_dice}")

            # calculate the iou
            pixel_accuracy = metrics.calculate_pixel_wise_accuracy(outputs, binary_mask)
            val_pixel_accuracy += pixel_accuracy
            # print(f"Validation Pixel-wise Accuracy: {val_pixel_accuracy}")

            # calculate the sensitivity
            recall = metrics.calculate_recall(outputs.cpu().detach().numpy(), masks.cpu().detach().numpy())
            val_recall += recall

            # Print the accuracy
            # print(f'Accuracy: {accuracy}')



        # Print the accuracy
        val_accuracy = (val_accuracy / val_steps) * 100
        val_loss = val_loss / val_steps
        val_iou = train_iou / val_steps
        val_dice = train_dice / val_steps
        val_pixel_accuracy = train_pixel_accuracy / val_steps
        val_recall = val_recall / val_steps


        print("Training accuracy: {:.2f}%, Validation accuracy: {:.2f}%, Traning Loss: {:.4f}, Validation Loss: {:.4f}, Traning Sensitivity: {:.4f}, Validation Sensitivity: {:.4f}, ".format(train_accuracy, val_accuracy, train_loss, val_loss, train_recall, val_recall))
        print("Training iou: {:.4f}, Validation iou: {:.4f}, Traning dice: {:.4f}, Validation dice: {:.4f}, Traning pixel accuracy: {:.4f}, Validation pixel accuracy: {:.4f}".format(train_iou, val_iou, train_dice, val_dice, train_pixel_accuracy, val_pixel_accuracy))

        # update our training history
        H["train_accuracy"].append(train_accuracy)
        H["val_accuracy"].append(val_accuracy)
        H["train_loss"].append(train_loss)
        H["val_loss"].append(val_loss)
        H["train_iou"].append(train_iou)
        H["val_iou"].append(val_iou)
        H["train_dice"].append(train_dice)
        H["val_dice"].append(val_dice)
        H["train_pixel_accuracy"].append(train_pixel_accuracy)
        H["val_pixel_accuracy"].append(val_pixel_accuracy)

        # Save the model
        torch.save(model.state_dict(), 'model.pt')


    # #plotting graphs
    # # plot the training & validation accuracy
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(H["train_accuracy"], label="train_accuracy")
    # plt.plot(H["val_accuracy"], label="val_accuracy")
    # plt.title("Training & Validation Accuracy on Dataset")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Accuracy (%)")
    # plt.legend(loc="lower left")
    # plt.savefig(config.ACCURACY_PLOT_PATH)

        # # plot the training & validation loss
        # plt.style.use("ggplot")
        # plt.figure()
        # plt.plot(H["train_loss"], label="train_loss")
        # plt.plot(H["val_loss"], label="val_loss")
        # plt.title("Training & Validation Loss on Dataset")
        # plt.xlabel("Epoch #")
        # plt.ylabel("Loss")
        # plt.legend(loc="lower left")
        # plt.savefig(config.LOSSES_PLOT_PATH)
        #
        # plot the training & validation iou
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_iou"], label="train_iou")
        plt.plot(H["val_iou"], label="val_iou")
        plt.title("Training & Validation IoU on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("iou")
        plt.legend(loc="lower left")
        plt.savefig(config.IOU_PLOT_PATH)

        # plot the training & validation dice coefficient
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_dice"], label="train_dice")
        plt.plot(H["val_dice"], label="val_dice")
        plt.title("Training & Validation Dice Coefficient on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("dice coefficient")
        plt.legend(loc="lower left")
        plt.savefig(config.DICE_PLOT_PATH)
        #
        # # plot the training & validation pixel accuracy
        # plt.style.use("ggplot")
        # plt.figure()
        # plt.plot(H["train_pixel_accuracy"], label="train_pixel_accuracy")
        # plt.plot(H["val_pixel_accuracy"], label="val_pixel_accuracy")
        # plt.title("Training & Validation Pixel Accuracy on Dataset")
        # plt.xlabel("Epoch #")
        # plt.ylabel("pixel_accuracy")
        # plt.legend(loc="lower left")
        # plt.savefig(config.PIXEL_ACCURACY_PLOT_PATH)
        #

        # plot the training & validation dice coefficient
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_recall"], label="train_sen")
        plt.plot(H["val_recall"], label="val_sen")
        plt.title("Training & Validation Sensitivity on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Sensitivity")
        plt.legend(loc="lower left")
        plt.savefig(config.Recall_PLOT_PATH)
