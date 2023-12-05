"""
the main script that initialize the model
Author: Abdelkarim eljandoubi
date: Nov 2023
"""

from transformers import AutoModelForImageClassification, AutoConfig
from peft import LoraConfig, get_peft_model


def model_from_checkpoint(model_checkpoint,
                          r=16,
                          lora_alpha=16,
                          lora_dropout=0.1,
                          bias="none",
                          ):
    """create the function that initiate a model for hyperparameters search"""

    def model_init(trial):
        """It returns the initial model"""

        # get the model configuration and set the number of labels to 2 (0-1)
        config = AutoConfig.from_pretrained(model_checkpoint, num_labels=2)

        # load the model
        model = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
        )

        if trial is not None:
            # sugests hyperparamets for lora
            r_ = trial.suggest_categorical("r", [2**i for i in range(3, 10)])
            lora_alpha_ = trial.suggest_float(
                "lora_alpha", 1e-2, 1e2, log=True)
            lora_dropout_ = trial.suggest_float("lora_dropout", 0, 0.5)
            bias_ = trial.suggest_categorical("bias", ["none", "lora_only"])

        else:
            # use the defaults
            r_ = r
            lora_alpha_ = lora_alpha
            lora_dropout_ = lora_dropout
            bias_ = bias

        # set the lora configuration
        lora_config = LoraConfig(
            r=r_,
            lora_alpha=lora_alpha_,
            lora_dropout=lora_dropout_,
            bias=bias_,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        )

        return get_peft_model(model, lora_config)

    # return the function that intiate the model for the search
    return model_init
