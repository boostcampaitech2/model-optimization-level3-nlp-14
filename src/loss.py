"""Custom loss for long tail problem.

- Author: Junghoon Kim
- Email: placidus36@gmail.com
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.common import get_label_counts


class CustomCriterion:
    """Custom Criterion."""

    def __init__(self, samples_per_cls, device, fp16=False, loss_type="softmax"):
        if not samples_per_cls:
            loss_type = "softmax"
        else:
            self.samples_per_cls = samples_per_cls
            self.frequency_per_cls = samples_per_cls / np.sum(samples_per_cls)
            self.no_of_classes = len(samples_per_cls)
        self.device = device
        self.fp16 = fp16

        if loss_type == "softmax":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == "logit_adjustment_loss":
            tau = 1.0
            self.logit_adj_val = (
                torch.tensor(tau * np.log(self.frequency_per_cls))
                .float()
                .to(self.device)
            )
            self.logit_adj_val = (
                self.logit_adj_val.half() if fp16 else self.logit_adj_val.float()
            )
            self.logit_adj_val = self.logit_adj_val.to(device)
            self.criterion = self.logit_adjustment_loss

    def __call__(self, logits, labels):
        """Call criterion."""
        return self.criterion(logits, labels)

    def logit_adjustment_loss(self, logits, labels):
        """Logit adjustment loss."""
        logits_adjusted = logits + self.logit_adj_val.repeat(labels.shape[0], 1)
        loss = F.cross_entropy(input=logits_adjusted, target=labels)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.weights = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weights)(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


def get_weights(data_path):
    """get class weights by scikit-learn way
    ref : https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    Args:
        data_path (str): dataset path

    Returns:
        numpy.ndarray: class weights
    """
    class_num = get_label_counts(os.path.join(data_path, "train"))
    n_samples = sum(class_num)
    bin_count = np.array(class_num)
    n_classes = len(class_num)
    weights = n_samples / (n_classes * bin_count)
    return weights


def get_loss(loss_fn, fp16, weight, device):
    if loss_fn == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_fn == 'CrossEntropy_Weight':
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(device, dtype=torch.half if fp16 else torch.float))
    elif loss_fn == 'FocalLoss_Weight':
        loss_fn = FocalLoss(weight=torch.tensor(weight).to(device, dtype=torch.half if fp16 else torch.float))
    elif loss_fn == 'ContrastiveLoss':
        loss_fn = ContrastiveLoss()
    return loss_fn