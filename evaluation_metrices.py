import numpy as np
import sys
import torch


class Evaluation_metrices:
    # @staticmethod
    def calculate_metrics(output, target):
        # print("output_shape", output.shape)
        # print("target_shape", target.shape)
        tp = tn = fp = fn = 0
        for i in range(len(output)):
            # print("output_shape", output[i].shape)
        # print()
        # print("output", output[1][0])
        # print("output", output[1][1])
        # print("target", target[1][0])
        # print("target", target[1][1])


            threshold = 0.5
            converted_output = (output[i] > threshold).to(torch.int)
            converted_target = (target[i] > threshold).to(torch.int)

            # print("converted_output", converted_output.shape)
            # print("converted_target", converted_target.shape)

            # torch.set_printoptions(profile="full")
            # print("converted_output", converted_output[1][0])
            # print("converted_target", converted_target[1][0])

            # Assuming converted_target is a PyTorch tensor
            # tensor_data = target[1][0].detach().numpy()  # Convert to NumPy array
            # print(np.unique(tensor_data))  # Get unique elements using NumPy's unique function
            # tensor_data = converted_target[1][0].detach().numpy()  # Convert to NumPy array
            # print(np.unique(tensor_data))


            tp += torch.sum((converted_output == 1) & (converted_target == 1))
            fp += torch.sum((converted_output == 1) & (converted_target == 0))
            tn += torch.sum((converted_output == 0) & (converted_target == 0))
            fn += torch.sum((converted_output == 0) & (converted_target == 1))

        pos = torch.sum(converted_target == 1)
        neg = torch.sum(converted_target == 0)
        #
        # print(pos, neg)
        # print(tp, " ", fp, " ", tn, " ", fn )


        return tp, tn, fp, fn


    def calclate_accuracy(tp, tn, fp, fn):
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        return accuracy


    def calculate_recall(tp, tn, fp, fn):
        recall = (tp) / (tp + fn)
        return recall


    def calculate_dice_coefficient(tp, tn, fp, fn):
        dice_coefficient = (2 * tp) / ((2 * tp) + fp + fn)
        return dice_coefficient

    # def calculate_iou(y_true, y_pred):
    #     smooth = 1.
    #     y_true_f = y_true.view(-1)
    #     y_pred_f = y_pred.view(-1)
    #     intersection = torch.sum(y_true_f * y_pred_f)
    #     score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    #     return score

    def calculate_iou(tp, tn, fp, fn):
        iou = (tp) / (tp + fp + fn)
        return iou

    def calculate_specificity(tp, tn, fp, fn):
        specificity = (tn) / (tn + fp)
        return specificity

    #
    #
    # def calculate_dice_coefficient(output, target, threshold=0.5):
    #     output = (output > threshold).int()
    #     target = (target > 0.5).astype(np.int32)
    #
    #     intersection = (output * target).sum((1, 2)) * 2.0  # Multiplication by 2 for Dice coefficient
    #     dice_coefficient = (intersection + 1e-7) / (output.sum((1, 2)) + target.sum((1, 2)) + 1e-7)
    #
    #     return dice_coefficient.mean()
    #

    # def calculate_recall(output, target, threshold=0.5):
    #     output = (output > threshold).astype(np.int32)
    #     target = (target > 0.5).astype(np.int32)
    #
    #     true_positive = np.sum((output == 1) & (target == 1))
    #     false_negative = np.sum((output == 0) & (target == 1))
    #
    #     recall = (true_positive + 1e-7) / ((true_positive + false_negative) + 1e-7)
    #
    #     return recall.mean()
    #
    # def calculate_iou(output, target, threshold=0.5):
    #     output = (output > threshold).int()
    #     target = (target > 0.5).astype(np.int32)
    #
    #     intersection = (output & target).float().sum((1, 2))  # Sum of intersection
    #     union = (output | target).float().sum((1, 2))       # Sum of union
    #
    #     iou = (intersection + 1e-7) / (union + 1e-7)        # Add a small value to avoid division by zero
    #
    #     return iou.mean()
    #
    # def calculate_dice_coefficient(output, target, threshold=0.5):
    #     output = (output > threshold).int()
    #     target = (target > 0.5).astype(np.int32)
    #
    #     intersection = (output * target).sum((1, 2)) * 2.0  # Multiplication by 2 for Dice coefficient
    #     dice_coefficient = (intersection + 1e-7) / (output.sum((1, 2)) + target.sum((1, 2)) + 1e-7)
    #
    #     return dice_coefficient.mean()
    #
    #
    # def calculate_pixel_wise_accuracy(output, target, threshold=0.5):
    #     output = (output > threshold).int()
    #     target = (target > 0.5).astype(np.int32)
    #
    #     correct_pixels = np.sum((output == target) * 1)
    #     total_pixels = output.size(1) * output.size(2)  # Total number of pixels per sample
    #
    #     accuracy = (correct_pixels + 1e-7) / (total_pixels + 1e-7)
    #
    #     return accuracy.mean()