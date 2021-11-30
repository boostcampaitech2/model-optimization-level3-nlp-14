"""Tune Model.
- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""
import os
import pickle
from datetime import datetime
import yaml
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.loss import CustomCriterion, get_weights, get_loss
from src.model import Efficientnet_b0
from src.utils.torch_utils import model_info, check_runtime
from src.utils.common import get_label_counts
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, List, Tuple
from optuna.pruners import HyperbandPruner
from subprocess import _args_from_interpreter_flags
import argparse
from optuna.integration.wandb import WeightsAndBiasesCallback # optuna>=v2.9.0

DATA_PATH = "/opt/ml/data"  # type your data path here that contains test, train and val directories
def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = 10 # trial.suggest_int("epochs", low=50, high=100, step=50)
    img_size = trial.suggest_categorical("img_size", [96, 112, 168, 224])
    n_select = trial.suggest_int("n_select", low=0, high=6, step=2)
    batch_size = trial.suggest_int("batch_size", low=16, high=32, step=16)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    init_lr = trial.suggest_float("init_lr", 1e-4, 1e-1,log=True)
    
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
        "OPTIMIZER" : optimizer_name,
        "INIT_LR" : init_lr,
    }


def objective(trial: optuna.trial.Trial, log_dir: str, device) -> Tuple[float, int, float]:
    """Optuna objective.
    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    hyperparams = search_hyperparam(trial)

    model = Efficientnet_b0()
    model_path = os.path.join(log_dir, "best.pt") # result model will be saved in this path
    print(f"Model save path: {model_path}")
    model.to(device)

    # check ./data_configs/data.yaml for config information
    data_config: Dict[str, Any] = {}
    data_config["DATA_PATH"] = DATA_PATH
    data_config["DATASET"] = "TACO"
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]
    data_config["AUG_TRAIN"] = "randaugment_train"
    data_config["AUG_TEST"] = "simple_augment_test"
    data_config["AUG_TRAIN_PARAMS"] = {
        "n_select": hyperparams["n_select"],
    }
    data_config["AUG_TEST_PARAMS"] = None
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["EPOCHS"] = hyperparams["EPOCHS"]
    data_config["VAL_RATIO"] = 0.2
    data_config["INIT_LR"] = hyperparams["INIT_LR"]
    data_config["FP16"] = True
    data_config["SUBSET_SAMPLING_RATIO"] = 0.5 # 0 means full data
    data_config["LOSS"] = 'CrossEntropy_Weight'

    trial.set_user_attr('hyperparams',  hyperparams)
    trial.set_user_attr('data_config', data_config)
    for key, value in trial.params.items():
        print(f"    {key}:{value}")

    mean_time = check_runtime(
        model,
        [3]+[224, 224],
        device,
    )
    model_info(model, verbose=True)
    train_loader, val_loader, test_loader = create_dataloader(data_config)

    weights = get_weights(data_config["DATA_PATH"])
    criterion = get_loss(data_config["LOSS"], data_config["FP16"], weight=weights, device=device)

    if hyperparams["OPTIMIZER"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams["INIT_LR"])
    else:
        optimizer = getattr(optim, hyperparams["OPTIMIZER"])(model.parameters(), lr=hyperparams["INIT_LR"])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hyperparams["INIT_LR"],
        steps_per_epoch=len(train_loader),
        epochs=hyperparams["EPOCHS"],
        pct_start=0.05,
        cycle_momentum=True if hyperparams["OPTIMIZER"] == "SGD" else False
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if data_config["FP16"] and device != torch.device("cpu") else None
    )

    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        model_path=model_path,
        device=device,
        verbose=1,
    )
    trainer.train(train_loader, hyperparams["EPOCHS"], val_dataloader=val_loader)
    loss, f1_score, acc_percent = trainer.test(model, test_dataloader=val_loader)
    params_nums = count_model_params(model)

    model_info(model, verbose=True)
    print('='*50)
    return f1_score, params_nums, mean_time


def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "f1_score",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : f1_score >= threshold
    threshold = 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def tune(gpu_id, storage: str = None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler()
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    log_dir = os.path.join("/opt/ml/code", os.path.join("exp", 'latest'))
    # log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest')) 
    log_dir_start = log_dir + '/best.pt'

    if os.path.exists(log_dir_start): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    wandb_kwargs = {"project": "optuna-backbone-ce", 'name': 'efficientnetb0'}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, metric_name=['value_0', 'value_1', 'value_2'])

    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name="automl101",
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, log_dir, device), n_trials=100, callbacks=[wandbc], n_jobs=1)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}, value3:{tr.values[2]}")
        save_path = os.path.join(log_dir, f'trial_id-{str(tr._trial_id).zfill(4)}')
        os.makedirs(save_path, exist_ok=True)
        tr_info = os.path.join(save_path, f"f1_{tr.values[0]:.4f}-n_params_{tr.values[1]}-time_{tr.values[2]:.4f}.txt")
        with open(tr_info, "w") as f:
            f.write(f"""f1:{tr.values[0]}
            n_params:{tr.values[1]}
            time:{tr.values[2]}
            """)
        with open(os.path.join(save_path, "hyperparams.yml"), "w") as f:
            yaml.dump(tr.user_attrs['hyperparams'], f, default_flow_style=None, sort_keys=False)
        with open(os.path.join(save_path, "data.yml"), "w") as f:
            yaml.dump(tr.user_attrs['data_config'], f, default_flow_style=None, sort_keys=False)
        
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    with open(os.path.join(log_dir, 'study.pkl'), 'wb') as f:
        pickle.dump(study, f)
    best_trial = get_best_trial_with_condition(study)
    print(best_trial)
    with open(os.path.join(log_dir, 'best_trials.pkl'), 'wb') as f:
        pickle.dump(best_trials, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="sqlite:///exp.db", type=str, help="Optuna database storage path.")
    args = parser.parse_args()
    tune(args.gpu, storage=args.storage if args.storage != "" else None)
