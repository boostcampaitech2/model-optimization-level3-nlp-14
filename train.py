"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import wandb

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info


def train(
    hyperparams: Dict[str, Any],
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "hyperparams.yml"), "w") as f:
        yaml.dump(hyperparams, f, default_flow_style=False)
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(
            torch.load(model_path, map_location=device)
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
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    wandb.watch(model_instance.model, criterion, log='all', log_freq=10)
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/mobilenetv3.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="epochs"
    )
    args = parser.parse_args()

    # Load yml file from exp/latest/trial_id-####
    hyperparams = read_yaml(cfg=os.path.join(args.trial_dir, 'hyperparams.yml'))
    model_config = read_yaml(cfg=os.path.join(args.trial_dir, 'model.yml'))
    data_config = read_yaml(cfg=os.path.join(args.trial_dir, 'data.yml'))

    # Modify hyperparameter for training
    data_config["EPOCHS"] = args.epochs
    data_config["SUBSET_SAMPLING_RATIO"] = 0
    data_config["AUG_TRAIN_PARAMS"] = {"n_select" : 6}
    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp_train", 'latest'))

    if os.path.exists(log_dir): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    wandb.init(project="optuna-test-du-best_trials", name='trial_0073-rs42-randaug2-resume_test')
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
        resume=True,
        fp16=data_config["FP16"],
        device=device,
    )