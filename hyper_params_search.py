"""
the main script that do hyper parameter search
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
import json
import os
from set_trainer import lora_trainer


def hp_space(trial):
    """define the hyperparameter search space"""

    return {"num_train_epochs": trial.suggest_int("num_train_epochs", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3,
                                                 log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [2**i for i in range(5, 8)]),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [2**i for i in range(5)])
            }


def search(model_checkpoint, n_trials):
    """execute the search"""

    # check if this step has been executed
    if os.path.isfile("optimal.json"):
        return

    # load the trainer
    trainer = lora_trainer(model_checkpoint)

    # search for the best hyperparameters
    best_run = trainer.hyperparameter_search(
        n_trials=n_trials, hp_space=hp_space)

    # write the best hyperparameters
    with open("optimal.json", "w", encoding='utf-8') as fp:
        json.dump(best_run.hyperparameters, fp)
