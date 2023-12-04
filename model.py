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
                          ) :
    
    """create the function that initiate a model for hyperparameters search"""

    def model_init(trial):
        """It returns the initial model"""
        
        model = AutoModelForImageClassification.from_pretrained(
          model_checkpoint,
          ignore_mismatched_sizes=True,
          )
        
        if trial is not None:
            r_ = trial.suggest_categorical("r", [2**i for i in range(3,7)])
            lora_alpha_ = trial.suggest_float("lora_alpha", 1e-2, 1e2, log=True)
            lora_dropout_ = trial.suggest_float("lora_dropout", 0, 0.5)
            bias_ = trial.suggest_categorical("bias", ["none","all","lora_only"])
            
        else:
            r_=r
            lora_alpha_=lora_alpha
            lora_dropout_=lora_dropout
            bias_=bias
            
    
        lora_config = LoraConfig(
        r=r_,
        lora_alpha=lora_alpha_,
        lora_dropout=lora_dropout_,
        bias=bias_,
        target_modules=["query","key", "value"],
        modules_to_save=["classifier"],
        )
        
        return get_peft_model(model, lora_config)
  
    return model_init