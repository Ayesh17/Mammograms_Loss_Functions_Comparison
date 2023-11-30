import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from pathlib import Path

import config
from dataset import SegmentationDataset
from evaluation_metrices import Evaluation_metrices

from AUNet_1 import AUNet_R16
from main import convert_to_binary

# Assuming you have defined your SegmentationDataset class and config variables

# Data reading and splitting
all_image_npy_paths = sorted(Path(config.IMAGE_DATASET_PATH).glob("*.npy"))
all_mask_npy_paths = sorted(Path(config.MASK_DATASET_PATH).glob("*.npy"))

train_images, val_images = train_test_split(all_image_npy_paths, test_size=0.2, random_state=42)
train_masks, val_masks = train_test_split(all_mask_npy_paths, test_size=0.2, random_state=42)

train_dataset = SegmentationDataset(train_images, train_masks)
val_dataset = SegmentationDataset(val_images, val_masks)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False)

# Model Initialization
model = AUNet_R16()

# Loss Function and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.INIT_LR)


# initialize a dictionary to store training history
H = {"train_accuracy": [], "val_accuracy": [], "train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [], "train_dice": [], "val_dice": [], "train_pixel_accuracy": [], "val_pixel_accuracy": [] , "train_recall": [], "val_recall": [] }


# Training Loop
for epoch in range(config.EPOCHS):
    model.train()
    train_loss = 0.0

    # print(".............Training.............")
    # Train loop
    model.train()
    train_accuracy = 0
    train_iou = 0
    train_dice = 0
    train_pixel_accuracy = 0
    train_recall = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.float()
        masks = masks.float()
        masks = (masks - masks.min()) / (masks.max() - masks.min())

        images = images.to(torch.float).view(16, 3, 256, 256)
        masks = masks.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        # outputs = F.interpolate(outputs, size=(256, 256), mode='nearest')

        outputs = torch.sigmoid(outputs)
        # print("outputs", torch.min(outputs), torch.max(outputs))

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate the accuracy
        accuracy = (outputs > 0.5).float().mean()
        train_accuracy += accuracy

        # convert the mask to binary
        binary_mask = convert_to_binary(masks)

        # calculate the iou
        iou = Evaluation_metrices.calculate_iou(outputs, binary_mask)
        train_iou += iou
        # print(f"Training IoU: {train_iou}")

        # calculate the dice_score
        dice = Evaluation_metrices.calculate_dice_coefficient(outputs, binary_mask)
        train_dice += dice
        # print(f"Training Dice Coefficient: {train_dice}")

        # calculate the pixel_wise_accuracy
        pixel_accuracy = Evaluation_metrices.calculate_pixel_wise_accuracy(outputs, binary_mask)
        train_pixel_accuracy += pixel_accuracy
        # print(f"Training Pixel-wise Accuracy: {train_pixel_accuracy}")

        # calculate the sensitivity
        recall = Evaluation_metrices.calculate_recall(outputs.cpu().detach().numpy(), masks.cpu().detach().numpy())
        train_recall += recall

    train_steps = len(train_loader)
    train_loss /= len(train_loader)
    train_accuracy = (train_accuracy / train_steps) * 100
    train_iou = train_iou / train_steps
    train_dice = train_dice / train_steps
    train_pixel_accuracy = train_pixel_accuracy / train_steps
    train_recall = train_recall / train_steps

    # Evaluation Loop
    model.eval()
    val_loss = 0.0
    model.eval()
    val_accuracy = 0
    val_iou = 0
    val_dice = 0
    val_pixel_accuracy = 0
    val_recall = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.float()
            masks = masks.float()
            masks = (masks - masks.min()) / (masks.max() - masks.min())

            images = images.to(torch.float).view(8, 3, 256, 256)
            masks = masks.unsqueeze(1)

            outputs = model(images)

            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Calculate the accuracy
            accuracy = (outputs > 0.5).float().mean()
            val_accuracy += accuracy

            # convert the mask to binary
            binary_mask = convert_to_binary(masks)

            # calculate the iou
            iou = Evaluation_metrices.calculate_iou(outputs, binary_mask) * 100
            val_iou += iou
            # print(f"Validation IoU: {val_iou}")

            # calculate the iou
            dice = Evaluation_metrices.calculate_dice_coefficient(outputs, binary_mask)
            val_dice += dice
            # print(f"Validation Dice Coefficient: {val_dice}")

            # calculate the iou
            pixel_accuracy = Evaluation_metrices.calculate_pixel_wise_accuracy(outputs, binary_mask)
            val_pixel_accuracy += pixel_accuracy
            # print(f"Validation Pixel-wise Accuracy: {val_pixel_accuracy}")

            # calculate the sensitivity
            recall = Evaluation_metrices.calculate_recall(outputs.cpu().detach().numpy(), masks.cpu().detach().numpy())
            val_recall += recall

        val_steps = len(val_loader)
        val_loss /= len(val_loader)
        val_accuracy = (val_accuracy / val_steps) * 100
        val_iou = train_iou / val_steps
        val_dice = train_dice / val_steps
        val_pixel_accuracy = train_pixel_accuracy / val_steps
        val_recall = val_recall / val_steps

    print(
        "Training accuracy: {:.2f}%, Validation accuracy: {:.2f}%, Traning Loss: {:.4f}, Validation Loss: {:.4f}, Traning Sensitivity: {:.4f}, Validation Sensitivity: {:.4f}, ".format(
            train_accuracy, val_accuracy, train_loss, val_loss, train_recall, val_recall))
    print(
        "Training iou: {:.4f}, Validation iou: {:.4f}, Traning dice: {:.4f}, Validation dice: {:.4f}, Traning pixel accuracy: {:.4f}, Validation pixel accuracy: {:.4f}".format(
            train_iou, val_iou, train_dice, val_dice, train_pixel_accuracy, val_pixel_accuracy))

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

# Save the trained model
torch.save(model.state_dict(), 'AUNet_R16_trained.pth')

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

# Evaluation Metrics Calculation
# Here, you can add evaluation metrics such as IOU, Dice Coefficient, etc. on the validation set
# Evaluate the model on validation data using appropriate metrics


