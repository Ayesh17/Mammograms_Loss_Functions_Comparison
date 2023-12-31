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





# Load the mammogram images and masks path
all_image_npy_paths = sorted(Path(config.IMAGE_DATASET_PATH).glob("*.npy"))
all_mask_npy_paths = sorted(Path(config.MASK_DATASET_PATH).glob("*.npy"))


# Define lists to store evaluation metrics across folds
all_train_loss, all_val_loss = [], []
all_train_accuracy, all_val_accuracy = [], []
all_train_iou, all_val_iou = [], []
all_train_dice, all_val_dice = [], []
all_train_specificity, all_val_specificity = [], []
all_train_recall, all_val_recall = [], []


# Define the number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


for fold, (train_idx, val_idx) in enumerate(kf.split(all_image_npy_paths)):
    print(f"Fold {fold + 1}")

    # Split dataset into train and validation for this fold
    train_images = [all_image_npy_paths[i] for i in train_idx]
    val_images = [all_image_npy_paths[i] for i in val_idx]
    train_masks = [all_mask_npy_paths[i] for i in train_idx]
    val_masks = [all_mask_npy_paths[i] for i in val_idx]

    # Create the training and validation datasets for this fold
    train_dataset = SegmentationDataset(train_images, train_masks)
    val_dataset = SegmentationDataset(val_images, val_masks)

    # Create the data loaders for this fold
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False)


    print(f"[INFO] found {len(train_dataset)} examples in the training set...")
    print(f"[INFO] found {len(val_dataset)} examples in the test set...")


    # calculate steps per epoch for training and test set
    train_steps = len(train_dataset) // config.TRAIN_BATCH_SIZE
    val_steps = len(val_dataset) // config.TEST_BATCH_SIZE

    # Create the model
    # model = AUNet_R16()
    model = UNet()
    # model = build_unet()

    # Define the loss function
    # loss_function = nn.BCELoss()
    # loss_function = nn.CrossEntropyLoss()
    loss_function = Semantic_loss_functions()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.Learning_rate)

    #evaluation metrices
    metrics = Evaluation_metrices

    # initialize a dictionary to store training history
    H = {"train_accuracy": [], "val_accuracy": [], "train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [], "train_dice": [], "val_dice": [], "train_specificity": [], "val_specificity": [] , "train_recall": [], "val_recall": [] }

    min_valid_loss = np.inf

    # Train the model
    for epoch in range(config.EPOCHS):
        if epoch == 39:
            config.Learning_rate = 0.00005
        elif epoch == 69:
            config.Learning_rate = 0.00001
        elif epoch == 99:
            config.Learning_rate = 0.000001
        print("Epoch :", epoch+1, config.Learning_rate)

        # print(".............Training.............")
        # Train loop
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_iou = 0
        train_dice = 0
        train_specificity = 0
        train_recall = 0
        # train_loop = enumerate(tqdm.tqdm(train_loader, total=len(train_loader), leave=True))
        for images, masks in train_loader:
            images = images.float()

            # # convert CBIS to RGB
            # binary_image = np.expand_dims(images, axis=-1)
            # # Stack the single-channel array to create an RGB image by replicating the channel
            # rgb_image = np.concatenate([binary_image, binary_image, binary_image], axis=-1)
            # images = rgb_image

            masks = masks.float()

            # print(masks)
            # print("pre_masks", masks.max(), masks.min())
            masks = (masks - masks.min()) / (masks.max() - masks.min())
            # print("masks", masks.max(), masks.min() )
            # print(masks)

            # print("images_training", images.shape)
            # Resize the target tensor to match the shape of the input tensor
            images_tensor = torch.as_tensor(images, dtype=torch.float32).clone().detach()
            images = images_tensor.view(4, 3, 256, 256)
            masks = masks.unsqueeze(1)

            # Forward pass
            outputs = model(images)

            # Assuming 'outputs' is the tensor representing the predicted segmentation mask
            outputs = torch.sigmoid(outputs)

            # Calculate the loss
            # loss = loss_function(outputs, masks)
            # print("Mean Loss:", loss.item())

            # print("outputs",torch.min(outputs), torch.max(outputs))
            # print("masks",torch.min(masks), torch.max(masks))
            loss = loss_function.bce_dice_loss(outputs, masks)
            train_loss += loss


            tp, tn, fp, fn = metrics.calculate_metrics(outputs, masks)
            # print("aaaaa")
            # print("metrics", tp, tn, fp, fn)

            # calculate accuracy
            accuracy = metrics. calclate_accuracy(tp, tn, fp, fn)
            # print("accuracy", accuracy)
            train_accuracy += accuracy

            # calculate accuracy
            recall = metrics.calculate_recall(tp, tn, fp, fn)
            # print("recall", recall)
            train_recall += recall

            # calculate accuracy
            dice_coefficient = metrics.calculate_dice_coefficient(tp, tn, fp, fn)
            # print("dice_coefficient", dice_coefficient)
            train_dice += dice_coefficient


            # calculate the iou
            iou = metrics.calculate_iou(masks, outputs)
            train_iou += iou
            # print(f"Training IoU: {train_iou}")


            # calculate the specificity
            specificity = metrics.calculate_specificity(tp, tn, fp, fn)
            train_specificity += specificity
            # print(f"Training specificity: {train_specificity}")

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
        train_specificity = train_specificity / train_steps
        train_recall = train_recall / train_steps


        # print("..................Validation..............")
        # Validation loop
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_iou = 0
        val_dice = 0
        val_specificity = 0
        val_recall = 0
        with torch.no_grad():
            # val_loop = enumerate(tqdm.tqdm(val_loader, total=len(val_loader), leave=True))
            for images, masks in val_loader:

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
                # images_tensor = torch.tensor(images, dtype=torch.float32)
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
                val_loss += loss
                loss = loss_function.bce_dice_loss(outputs, masks)



                # calculate accuracy
                accuracy = metrics.calclate_accuracy(tp, tn, fp, fn)
                # print("accuracy", accuracy)
                val_accuracy += accuracy

                # calculate accuracy
                recall = metrics.calculate_recall(tp, tn, fp, fn)
                # print("recall", recall)
                val_recall += recall

                # calculate accuracy
                dice_coefficient = metrics.calculate_dice_coefficient(tp, tn, fp, fn)
                # print("dice_coefficient", dice_coefficient)
                val_dice += dice_coefficient

                # calculate the iou
                iou = metrics.calculate_iou(masks, outputs)
                val_iou += iou
                # print(f"Validation IoU: {val_iou}")

                # calculate the iou
                specificity = metrics.calculate_specificity(tp, tn, fp, fn)
                val_specificity += specificity
                # print(f"Validation specificity: {val_specificity}")


            # Print the accuracy
            val_accuracy = (val_accuracy / val_steps) * 100
            val_loss = val_loss / val_steps
            val_iou = val_iou / val_steps
            val_dice = val_dice / val_steps
            val_specificity = val_specificity / val_steps
            val_recall = val_recall / val_steps


            print("Training accuracy: {:.2f}%, Validation accuracy: {:.2f}%, Traning Loss: {:.4f}, Validation Loss: {:.4f}, Traning Sensitivity: {:.4f}, Validation Sensitivity: {:.4f}, ".format(train_accuracy, val_accuracy, train_loss, val_loss, train_recall, val_recall))
            print("Training iou: {:.4f}, Validation iou: {:.4f}, Traning dice: {:.4f}, Validation dice: {:.4f}, Traning specificity: {:.4f}, Validation specificity: {:.4f}".format(train_iou, val_iou, train_dice, val_dice, train_specificity, val_specificity))

            # update our training history
            H["train_accuracy"].append(train_accuracy)
            H["val_accuracy"].append(val_accuracy)
            H["train_loss"].append(train_loss)
            H["val_loss"].append(val_loss)
            H["train_iou"].append(train_iou)
            H["val_iou"].append(val_iou)
            H["train_dice"].append(train_dice)
            H["val_dice"].append(val_dice)
            H["train_specificity"].append(train_specificity)
            H["val_specificity"].append(val_specificity)
            H["train_recall"].append(train_recall)
            H["val_recall"].append(val_recall)

            # Save the model
            if min_valid_loss > val_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
                min_valid_loss = val_loss
                # Saving State Dict
                torch.save(model.state_dict(), 'model.pt')

    # Record evaluation metrics at the end of each fold
    all_train_loss.append(H["train_loss"][-1].detach().numpy())
    all_val_loss.append(H["val_loss"][-1].detach().numpy())

    all_train_accuracy.append(H["train_accuracy"][-1].detach().numpy())
    all_val_accuracy.append(H["val_accuracy"][-1].detach().numpy())

    all_train_iou.append(H["train_iou"][-1].detach().numpy())
    all_val_iou.append(H["val_iou"][-1].detach().numpy())

    all_train_dice.append(H["train_dice"][-1].detach().numpy())
    all_val_dice.append(H["val_dice"][-1].detach().numpy())

    all_train_specificity.append(H["train_specificity"][-1].detach().numpy())
    all_val_specificity.append(H["val_specificity"][-1].detach().numpy())

    all_train_recall.append(H["train_recall"][-1].detach().numpy())
    all_val_recall.append(H["val_recall"][-1].detach().numpy())



# Calculate the mean of evaluation metrics across all folds
mean_train_loss = np.mean(all_train_loss)
mean_val_loss = np.mean(all_val_loss)

mean_train_accuracy = np.mean(all_train_accuracy)
mean_val_accuracy = np.mean(all_val_accuracy)

mean_train_iou = np.mean(all_train_iou)
mean_val_iou = np.mean(all_val_iou)

mean_train_dice = np.mean(all_train_dice)
mean_val_dice = np.mean(all_val_dice)

mean_train_specificity = np.mean(all_train_specificity)
mean_val_specificity = np.mean(all_val_specificity)

mean_train_recall = np.mean(all_train_recall)
mean_val_recall = np.mean(all_val_recall)

# Display mean evaluation metrics across all folds
print("Mean Training Loss:", mean_train_loss)
print("Mean Validation Loss:", mean_val_loss)

print("Mean Training Accuracy:", mean_train_accuracy)
print("Mean Validation Accuracy:", mean_val_accuracy)

print("Mean Training IoU:", mean_train_iou)
print("Mean Validation IoU:", mean_val_iou)

print("Mean Training Dice Coefficient:", mean_train_dice)
print("Mean Validation Dice Coefficient:", mean_val_dice)

print("Mean Training Specificity:", mean_train_specificity)
print("Mean Validation Specificity:", mean_val_specificity)

print("Mean Training Recall:", mean_train_recall)
print("Mean Validation Recall:", mean_val_recall)


# plotting graphs
# plot the training & validation accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_accuracy"], label="train_accuracy")
plt.plot(H["val_accuracy"], label="val_accuracy")
plt.title("Training & Validation Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy (%)")
plt.legend(loc="lower left")
plt.savefig(config.ACCURACY_PLOT_PATH)

# plot the training & validation loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"][-1].detach().numpy(), label="train_loss")
plt.plot(H["val_loss"][-1].detach().numpy(), label="val_loss")
plt.title("Training & Validation Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.LOSSES_PLOT_PATH)

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

# plot the training & validation specificity
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_specificity"], label="train_specificity")
plt.plot(H["val_specificity"], label="val_specificity")
plt.title("Training & Validation specificity on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("specificity")
plt.legend(loc="lower left")
plt.savefig(config.SPECIFICTY_PLOT_PATH)

# plot the training & validation sensitivity
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_recall"], label="train_recall")
plt.plot(H["val_recall"], label="val_recall")
plt.title("Training & Validation Sensitivity on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Sensitivity")
plt.legend(loc="lower left")
plt.savefig(config.RECALL_PLOT_PATH)
