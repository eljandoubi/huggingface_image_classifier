"""
the main script that trains the model
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
import json
from set_trainer import lora_trainer

def train_best(model_checkpoint:str)->None:
    
    """train a model with the optimal hyperparameters"""
    
    with open("optimal.json", 'r', encoding='utf-8') as file:
        hyperparameters= json.load(file)
        
    trainer = lora_trainer(model_checkpoint,mode="train",**hyperparameters)
    
    trainer.train()
    
    model_name = model_checkpoint.split("/")[-1]
    
    trainer.save_model(f"best-{model_name}-finetuned-lora")