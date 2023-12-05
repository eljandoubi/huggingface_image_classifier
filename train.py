"""
the main script that trains the model
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
import json
import os
from set_trainer import lora_trainer


def train_best(model_checkpoint):
    """train a model with the optimal hyperparameters"""

    # get the model name
    model_name = model_checkpoint.split("/")[-1]

    # set the output directory name
    dir_name = f"best-{model_name}-finetuned-lora"

    # check if the model has already been trained
    if os.path.isdir(dir_name):
        return

    # load best hyperparameters
    with open("optimal.json", 'r', encoding='utf-8') as file:
        hyperparameters = json.load(file)

    # load the trainer
    trainer = lora_trainer(model_checkpoint, **hyperparameters)

    # train the model
    trainer.train()

    # save the model
    trainer.save_model(dir_name)
