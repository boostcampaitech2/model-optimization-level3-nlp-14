"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
import yaml
import random
import pickle
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from src.dataloader import create_dataloader
from src.loss import get_weights, get_loss
from src.model import Efficientnet_b0, Model
from src.trainer import TorchTrainer, KD_TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info


def set_seed(seed: int = 42):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


def train(
    hyperparams: Dict[str, Any],
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    new_log_dir: str,
    resume: bool,
    fp16: bool,
    device: torch.device,
    distillation : bool,
) -> Tuple[float, float, float]:
    """Train."""
    
    set_seed(seed=42)
    
    # Save RNG state in non-distributed training
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    print("RNG_STATES: ", rng_states)

    if torch.cuda.is_available():
        # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
        rng_states["cuda"] = torch.cuda.random.get_rng_state_all()

    with open(os.path.join(log_dir, "rng_state.pkl"), "wb") as f:
        pickle.dump(data_config, f)

    # save model_config, data_config
    with open(os.path.join(log_dir, "hyperparams.yml"), "w") as f:
        yaml.dump(hyperparams, f, default_flow_style=False)
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_path = os.path.join(log_dir, "best.pt")
    resume_model_path = os.path.join(new_log_dir, "best.pt")

    model_instance = Model(model_config, verbose=True)
    if distillation == True:
        teacher_model = Efficientnet_b0()
        teacher_model.load_state_dict(torch.load("/opt/ml/code/exp/latest/teacher_model.pt"))
        teacher_model.to(device)

    print(f"Model save path: {model_path}")
    if os.path.isfile(resume_model_path) and resume:
        print("Resume Training from ", resume_model_path)
        model_instance.model.load_state_dict(
            torch.load(resume_model_path, map_location=device)
        )
    model_instance.model.to(device)
    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    if hyperparams["OPTIMIZER"] == "SGD":
        optimizer = torch.optim.SGD(model_instance.model.parameters(), lr=hyperparams["INIT_LR"])
    else:
        optimizer = getattr(optim, hyperparams["OPTIMIZER"])(model_instance.model.parameters(), lr=hyperparams["INIT_LR"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    weights = get_weights(data_config["DATA_PATH"])
    criterion = get_loss(data_config["LOSS"], data_config["FP16"], weight=weights, device=device)
    wandb.watch(model_instance.model, criterion, log='all', log_freq=10)
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    if distillation:
        trainer = KD_TorchTrainer(
        student_model=model_instance.model,
        teacher_model=teacher_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    else:
        trainer = TorchTrainer(
            model=model_instance.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            model_path=model_path,
            verbose=1,
        )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
        wandb_log=True,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc

def backbone_train(
    hyperparams: Dict[str, Any],
    model,
    data_config: Dict[str, Any],
    log_dir: str,
    new_log_dir: str,
    resume: bool,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train for pretrained model"""
    
    set_seed(seed=42)
    
    # Save RNG state in non-distributed training
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    print("RNG_STATES: ", rng_states)

    if torch.cuda.is_available():
        # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
        rng_states["cuda"] = torch.cuda.random.get_rng_state_all()

    with open(os.path.join(log_dir, "rng_state.pkl"), "wb") as f:
        pickle.dump(data_config, f)

    # save model_config, data_config
    with open(os.path.join(log_dir, "hyperparams.yml"), "w") as f:
        yaml.dump(hyperparams, f, default_flow_style=False)
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    model = model
    model_path = os.path.join(log_dir, "best.pt")
    resume_model_path = os.path.join(new_log_dir, "best.pt")

    print(f"Model save path: {model_path}")
    if os.path.isfile(resume_model_path) and resume:
        print("Resume Training from ", resume_model_path)
        model.load_state_dict(
            torch.load(resume_model_path, map_location=device)
        )
    model.to(device)
    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    if hyperparams["OPTIMIZER"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams["INIT_LR"])
    else:
        optimizer = getattr(optim, hyperparams["OPTIMIZER"])(model.parameters(), lr=hyperparams["INIT_LR"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    weights = get_weights(data_config["DATA_PATH"])
    criterion = get_loss(data_config["LOSS"], data_config["FP16"], weight=weights, device=device)
    wandb.watch(model, criterion, log='all', log_freq=10)
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
        wandb_log=True,
    )

    # evaluate model with test set
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--trial_dir",
        default="exp_train/latest",
        type=str,
        help="config dir",
    )
    parser.add_argument(
        "--epochs", default=1000, type=int, help="epochs"
    )
    parser.add_argument(
        "--backbone", default=False, type=bool, help="Whether to use pretrained model or not"
    )
    parser.add_argument(
        "--distillation", default=False, type=bool, help="Whether to apply Knowledge Distillation"
    )
    args = parser.parse_args()

    # Load yml file from exp/latest/trial_id-####
    hyperparams = read_yaml(cfg=os.path.join(args.trial_dir, 'hyperparams.yml'))
    if args.backbone == False:
        model_config = read_yaml(cfg=os.path.join(args.trial_dir, 'model.yml'))
    else:
        model = Efficientnet_b0()
    data_config = read_yaml(cfg=os.path.join(args.trial_dir, 'data.yml'))

    # Modify hyperparameter for training
    data_config["EPOCHS"] = args.epochs
    data_config["SUBSET_SAMPLING_RATIO"] = 0
    if args.distillation:
        data_config["LOSS"] = 'DistillationLoss_Weight'
    else:
        data_config["LOSS"] = 'CrossEntropy_Weight'
    data_config["AUG_TRAIN_PARAMS"] = {"n_select" : 6}
    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp_train", 'latest'))
    log_dir_start = log_dir + '/best.pt'

    if os.path.exists(log_dir_start): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)
    else:
        new_log_dir = ""

    os.makedirs(log_dir, exist_ok=True)

    wandb.init(project="optuna-wrapup", name='test')
    if args.backbone == False:
        wandb.config.update({
            'hyperparams' : hyperparams,
            'model_config' : model_config,
            'data_config' : data_config
            })

        test_loss, test_f1, test_acc = train(
        hyperparams=hyperparams,
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        new_log_dir=new_log_dir,
        resume=False,
        fp16=data_config["FP16"],
        device=device,
        distillation=args.distillation
    )
    else:
        wandb.config.update({
            'hyperparams' : hyperparams,
            'data_config' : data_config
            })

        test_loss, test_f1, test_acc = backbone_train(
            hyperparams=hyperparams,
            model=model,
            data_config=data_config,
            log_dir=log_dir,
            new_log_dir=new_log_dir,
            resume=False,
            fp16=data_config["FP16"],
            device=device,
        )