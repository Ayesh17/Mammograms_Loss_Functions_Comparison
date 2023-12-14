import numpy as np
import sys
import torch


class Evaluation_metrices:
    # @staticmethod
    def calculate_metrics(output, target):
        # print("output_shape", output.shape)
        # print("target_shape", target.shape)
        tp = tn = fp = fn = 0
        overall_accuracy = 0
        overall_recall = 0
        overall_specificity = 0
        overall_dice = 0
        overall_iou= 0

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


            tp = torch.sum((converted_output == 1) & (converted_target == 1))
            fp = torch.sum((converted_output == 1) & (converted_target == 0))
            tn = torch.sum((converted_output == 0) & (converted_target == 0))
            fn = torch.sum((converted_output == 0) & (converted_target == 1))

            # print(tp, " ", fp, " ", tn, " ", fn)

            accuracy = 0 if (tp + fp + tn + fn) == 0 else (tp + tn) / (tp + fp + tn + fn)
            recall = 0 if (tp + fn) == 0 else (tp) / (tp + fn)
            specificity = 0 if (tn + fp) == 0 else (tn) / (tn + fp)
            dice_coefficient = 0 if ((2 * tp) + fp + fn) == 0 else (2 * tp) / ((2 * tp) + fp + fn)
            iou = 0 if (tp + fp + fn) == 0 else (tp) / (tp + fp + fn)


            overall_accuracy += accuracy
            overall_recall += recall
            overall_specificity += specificity
            overall_dice += dice_coefficient
            overall_iou += iou



        pos = torch.sum(converted_target == 1)
        neg = torch.sum(converted_target == 0)
        #
        # print(pos, neg)
        # print(tp, " ", fp, " ", tn, " ", fn )


        return overall_accuracy, overall_recall, overall_specificity, overall_dice, overall_iou

    #
    # def calclate_accuracy(tp, tn, fp, fn):
    #     accuracy = (tp + tn) / (tp + fp + tn + fn)
    #     return accuracy
    #
    #
    # def calculate_recall(tp, tn, fp, fn):
    #     recall = (tp) / (tp + fn)
    #     return recall
    #
    #
    # def calculate_dice_coefficient(tp, tn, fp, fn):
    #     dice_coefficient = (2 * tp) / ((2 * tp) + fp + fn)
    #     return dice_coefficient
    #
    # # def calculate_iou(y_true, y_pred):
    # #     smooth = 1.
    # #     y_true_f = y_true.view(-1)
    # #     y_pred_f = y_pred.view(-1)
    # #     intersection = torch.sum(y_true_f * y_pred_f)
    # #     score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    # #     return score
    #
    # def calculate_iou(tp, tn, fp, fn):
    #     iou = (tp) / (tp + fp + fn)
    #     return iou
    #
    # def calculate_specificity(tp, tn, fp, fn):
    #     specificity = (tn) / (tn + fp)
    #     return specificity
