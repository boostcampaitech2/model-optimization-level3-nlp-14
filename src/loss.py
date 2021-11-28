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

    def __init__(self, samples_per_cls, device, fp16=False, loss_type="softmax", weights=None, focal=False):
        if not samples_per_cls:
            loss_type = "softmax"
        else:
            self.samples_per_cls = samples_per_cls
            self.frequency_per_cls = samples_per_cls / np.sum(samples_per_cls)
            self.no_of_classes = len(samples_per_cls)
            # max_class = np.max(self.samples_per_cls)
            # self.weights = (max_class / np.array(self.samples_per_cls))  
        if weights is not None:
            self.weights = weights
        if focal is not None:
            self.focal=False
        self.device = device
        self.fp16 = fp16

        if loss_type == "softmax" and self.weights is not None and self.focal:
            self.criterion = FocalLoss(weight=torch.tensor(self.weights).to(self.device, dtype=torch.half))
        elif loss_type == "softmax" and self.weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(self.device, dtype=torch.half))
        elif loss_type == "softmax":
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


def get_class_weights(data_path):
    class_num = get_label_counts(os.path.join(data_path, "train"))
    base_class = np.max(class_num)
    class_weight = (base_class / np.array(class_num))
    return class_weight