"""
the main script that initialize the model
Author: Abdelkarim eljandoubi
date: Nov 2023
"""

from transformers import AutoModelForImageClassification
from peft import LoraConfig, get_peft_model


def model_from_checkpoint(model_checkpoint,
          r = 16,
          lora_alpha = 16,
          lora_dropout = 0.1,
          bias = "none",
          )-> callable :
    
    """create the function that initiate a model for hyperparameters search"""

    def model_init(trial):
        """It returns the initial model"""
        
        model = AutoModelForImageClassification.from_pretrained(
          model_checkpoint,
          ignore_mismatched_sizes=True,
          )
        
        if trial is not None:
            r = trial.suggest_categorical("r", [2**i for i in range(3,10)])
            lora_alpha = trial.suggest_float("lora_alpha", 1e-3, 1e3, log=True)
            lora_dropout = trial.suggest_float("lora_dropout", 0, 0.9)
            bias = trial.suggest_categorical("bias", ["none","all","lora_only"])
            
    
        lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=lora_dropout,
        bias=bias,
        modules_to_save=["classifier"],
        )
        
        return get_peft_model(model, lora_config)
  
    return model_init