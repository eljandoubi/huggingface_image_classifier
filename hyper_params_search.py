"""
the main script that do hyper parameter search
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
import json
from set_trainer import lora_trainer

def hp_space(trial):
    return {"num_train_epochs": trial.suggest_int("num_train_epochs", 5, 50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3,
                                                 log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [2**i for i in range(10)]),
            "gradient_accumulation_steps": trial.suggest_int(
                "gradient_accumulation_steps", 1, 100)
           }

def search(model_checkpoint:str,n_trials:int)->None:
    """execute the search"""
    trainer = lora_trainer(model_checkpoint)
    best_run = trainer.hyperparameter_search(n_trials=n_trials, hp_space=hp_space)
    
    with open("optimal.json", "w", encoding='utf-8') as fp:
        json.dump(best_run.hyperparameters, fp)