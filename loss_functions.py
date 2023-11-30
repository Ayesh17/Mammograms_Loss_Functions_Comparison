import torch
import torch.nn.functional as F

beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1

class Semantic_loss_functions:
    def __init__(self):
        print("Semantic loss functions initialized")

    def dice_coef(self, y_true, y_pred):
        smooth = 1.
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        intersection = torch.sum(y_true_flat * y_pred_flat)
        return (2. * intersection + smooth) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + smooth)

    def sensitivity(self, y_true, y_pred):
        true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
        return true_positives / (possible_positives + 1e-5)

    def specificity(self, y_true, y_pred):
        true_negatives = torch.sum(torch.round(torch.clamp((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = torch.sum(torch.round(torch.clamp(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + 1e-5)

    def weighted_cross_entropyloss(self, y_true, y_pred):
        y_pred = torch.log(torch.clamp(y_pred, 1e-6, 1 - 1e-6))
        pos_weight = beta / (1 - beta)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=pos_weight)
        return torch.mean(loss)

    def convert_to_logits(self, y_pred):
        y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
        return torch.log(y_pred / (1 - y_pred))

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (torch.log1p(torch.exp(-torch.abs(logits))) + F.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
        logits = torch.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits, y_true, alpha, gamma, y_pred)

        return torch.mean(loss)

    def depth_softmax(self, matrix):
        sigmoid = lambda x: 1 / (1 + torch.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / torch.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss

    def bce_dice_loss(self, y_true, y_pred):
        loss = F.binary_cross_entropy(y_pred, y_true) + self.dice_loss(y_true, y_pred)
        return loss / 2.0

    def confusion(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = y_pred.clamp(0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = y_true.clamp(0, 1)
        y_neg = 1 - y_pos
        tp = torch.sum(y_pos * y_pred_pos)
        fp = torch.sum(y_neg * y_pred_pos)
        fn = torch.sum(y_pos * y_pred_neg)
        prec = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        return prec, recall

    def true_positive(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = torch.round(y_pred.clamp(0, 1))
        y_pos = torch.round(y_true.clamp(0, 1))
        tp = (torch.sum(y_pos * y_pred_pos) + smooth) / (torch.sum(y_pos) + smooth)
        return tp

    def true_negative(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = torch.round(y_pred.clamp(0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = torch.round(y_true.clamp(0, 1))
        y_neg = 1 - y_pos
        tn = (torch.sum(y_neg * y_pred_neg) + smooth) / (torch.sum(y_neg) + smooth)
        return tn

    def tversky_index(self, y_true, y_pred):
        y_true_pos = y_true.view(-1)
        y_pred_pos = y_pred.view(-1)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return (1 - pt_1).pow(gamma)

    def log_cosh_dice_loss(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

    def jacard_similarity(self, y_true, y_pred):
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)

        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
        return intersection / union

    def jacard_loss(self, y_true, y_pred):
        return 1 - self.jacard_similarity(y_true, y_pred)

    def ssim_loss(self, y_true, y_pred):
        return 1 - torch.image.ssim(y_true, y_pred, max_val=1)

    def unet3p_hybrid_loss(self, y_true, y_pred):
        focal_loss = self.focal_loss(y_true, y_pred)
        ms_ssim_loss = self.ssim_loss(y_true, y_pred)
        jacard_loss = self.jacard_loss(y_true, y_pred)
        return focal_loss + ms_ssim_loss + jacard_loss

    def basnet_hybrid_loss(self, y_true, y_pred):
        bce_loss = torch.nn.BCELoss()
        bce_loss = bce_loss(y_pred, y_true)

        ms_ssim_loss = self.ssim_loss(y_true, y_pred)
        jacard_loss = self.jacard_loss(y_true, y_pred)
        return bce_loss + ms_ssim_loss + jacard_loss