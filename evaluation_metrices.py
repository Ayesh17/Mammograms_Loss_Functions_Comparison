import numpy as np


class Evaluation_metrices:
    @staticmethod
    def calculate_recall(output, target, threshold=0.5):
        output = (output > threshold).astype(np.int32)
        target = (target > 0.5).astype(np.int32)

        true_positive = np.sum((output == 1) & (target == 1))
        false_negative = np.sum((output == 0) & (target == 1))

        recall = (true_positive + 1e-7) / ((true_positive + false_negative) + 1e-7)

        return recall.mean()

    def calculate_iou(output, target, threshold=0.5):
        output = (output > threshold).int()
        target = (target > 0.5).astype(np.int32)

        intersection = (output & target).float().sum((1, 2))  # Sum of intersection
        union = (output | target).float().sum((1, 2))       # Sum of union

        iou = (intersection + 1e-7) / (union + 1e-7)        # Add a small value to avoid division by zero

        return iou.mean()

    def calculate_dice_coefficient(output, target, threshold=0.5):
        output = (output > threshold).int()
        target = (target > 0.5).astype(np.int32)

        intersection = (output * target).sum((1, 2)) * 2.0  # Multiplication by 2 for Dice coefficient
        dice_coefficient = (intersection + 1e-7) / (output.sum((1, 2)) + target.sum((1, 2)) + 1e-7)

        return dice_coefficient.mean()


    def calculate_pixel_wise_accuracy(output, target, threshold=0.5):
        output = (output > threshold).int()
        target = (target > 0.5).astype(np.int32)

        correct_pixels = np.sum((output == target) * 1)
        total_pixels = output.size(1) * output.size(2)  # Total number of pixels per sample

        accuracy = (correct_pixels + 1e-7) / (total_pixels + 1e-7)

        return accuracy.mean()