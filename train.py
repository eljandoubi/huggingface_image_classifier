"""
the main script that trains the model
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
import json
import os
from set_trainer import lora_trainer

def train_best(model_checkpoint:str)->None:
    
    """train a model with the optimal hyperparameters"""
    
    
    model_name = model_checkpoint.split("/")[-1]
    
    dir_name = f"best-{model_name}-finetuned-lora"
    
    if os.path.isdir(dir_name):
        return
    
    with open("optimal.json", 'r', encoding='utf-8') as file:
        hyperparameters= json.load(file)
        
    trainer = lora_trainer(model_checkpoint,mode="train",**hyperparameters)
    
    trainer.train()
    
    
    trainer.save_model(dir_name)