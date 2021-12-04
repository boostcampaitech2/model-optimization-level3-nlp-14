"""PyTorch trainer module.

- Author: Jongkuk Lim, Junghoon Kim
- Contact: lim.jeikei@gmail.com, placidus36@gmail.com
"""

import os
import shutil
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from tqdm import tqdm
import wandb

from src.utils.torch_utils import EarlyStopping


def _get_n_data_from_dataloader(dataloader: DataLoader) -> int:
    """Get a number of data in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A number of data in dataloader
    """
    if isinstance(dataloader.sampler, SubsetRandomSampler):
        n_data = len(dataloader.sampler.indices)
    elif isinstance(dataloader.sampler, SequentialSampler):
        n_data = len(dataloader.sampler.data_source)
    else:
        n_data = len(dataloader) * dataloader.batch_size if dataloader.batch_size else 1

    return n_data


def _get_n_batch_from_dataloader(dataloader: DataLoader) -> int:
    """Get a batch number in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A batch number in dataloader
    """
    n_data = _get_n_data_from_dataloader(dataloader)
    n_batch = dataloader.batch_size if dataloader.batch_size else 1

    return n_data // n_batch


def _get_len_label_from_dataset(dataset: Dataset) -> int:
    """Get length of label from dataset.

    Args:
        dataset: torch dataset

    Returns:
        A number of label in set.
    """
    if isinstance(dataset, torchvision.datasets.ImageFolder) or isinstance(
        dataset, torchvision.datasets.vision.VisionDataset
    ):
        return len(dataset.classes)
    elif isinstance(dataset, torch.utils.data.Subset):
        return _get_len_label_from_dataset(dataset.dataset)
    else:
        raise NotImplementedError

class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        verbose: int = 1,
    ) -> None:
        """Initialize TorchTrainer class.
        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """

        self.model = model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device
        self.early_stopping = EarlyStopping(self.model_path, patience = 1000, delta=0)

    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
        wandb_log=False,
    ) -> Tuple[float, float]:
        """Train model.
        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation
        Returns:
            loss and accuracy
        """
        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        # Run training and track with wandb
        total_batches = len(train_dataloader) * n_epoch
        example_ct = 0  # number of examples seen
        batch_ct = 0
        
        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )

                example_ct += len(labels)
                batch_ct += 1
                if wandb_log and ((batch_ct + 1) % 100) == 0:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "loss" : running_loss / (batch + 1),
                            "acc" : correct / total,
                            "F1(macro)" : f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0)
                        }, step=example_ct)
                    # print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
            
            pbar.close()

            val_loss, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )
            if wandb_log:
                wandb.log(
                    {
                        "eval_loss" : val_loss,
                        "eval_acc" : test_acc,
                        "eval_F1(macro)" : test_f1
                    }, step=example_ct)
        
            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            self.early_stopping(val_loss, self.model, self.optimizer, self.scheduler)
            if self.early_stopping.early_stop:
                print("Early Stopping")
                return best_test_acc, best_test_f1

        return best_test_acc, best_test_f1



    @torch.no_grad()
    def test(
        self, model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.
        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)
        Returns:
            loss, f1, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = _get_len_label_from_dataset(test_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, (data, labels) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            running_loss += self.criterion(outputs, labels).item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy


class KD_TorchTrainer:
    """Pytorch Trainer for applying Knowledge Distillation"""

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        verbose: int = 1,
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device
        self.early_stopping = EarlyStopping(self.model_path, patience = 1000, delta=0)

    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
        wandb_log=False,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """
        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        # Run training and track with wandb
        total_batches = len(train_dataloader) * n_epoch
        example_ct = 0  # number of examples seen
        batch_ct = 0
        

        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            
            self.student_model.train()
            self.teacher_model.eval()
            
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        student_outputs = self.student_model(data)
                        teacher_outputs = self.teacher_model(data)
                else:
                    student_outputs = self.student_model(data)
                    teacher_outputs = self.teacher_model(data)
                student_outputs = torch.squeeze(student_outputs)
                loss = self.criterion(student_outputs, labels, teacher_outputs)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

                _, pred = torch.max(student_outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )

                example_ct += len(labels)
                batch_ct += 1
                if wandb_log and ((batch_ct + 1) % 100) == 0:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "loss" : running_loss / (batch + 1),
                            "acc" : correct / total,
                            "F1(macro)" : f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0)
                        }, step=example_ct)
                    # print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
            
            pbar.close()

            val_loss, test_f1, test_acc = self.test(
                student_model=self.student_model, test_dataloader=val_dataloader
            )
            if wandb_log:
                wandb.log(
                    {
                        "eval_loss" : val_loss,
                        "eval_acc" : test_acc,
                        "eval_F1(macro)" : test_f1
                    }, step=example_ct)
        
            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            self.early_stopping(val_loss, self.student_model, self.optimizer, self.scheduler)
            if self.early_stopping.early_stop:
                print("Early Stopping")
                return best_test_acc, best_test_f1

        return best_test_acc, best_test_f1

    @torch.no_grad()
    def test(
        self, student_model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = _get_len_label_from_dataset(test_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        student_model.to(self.device)
        student_model.eval()
        for batch, (data, labels) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = student_model(data)
            else:
                outputs = student_model(data)
            outputs = torch.squeeze(outputs)
            running_loss += F.cross_entropy(input=outputs, target=labels).item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy


def count_model_params(
    model: torch.nn.Module,
) -> int:
    """Count model's parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
