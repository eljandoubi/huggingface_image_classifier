"""
the main script that predicts labels
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
from transformers import AutoModelForImageClassification, pipeline
from peft import PeftModel
from load import load_data

def predict(model_checkpoint:str)->None:
    """Get prediction of the test data set"""
    
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        ignore_mismatched_sizes=True,
        )
    
    model_name = model_checkpoint.split("/")[-1]
    
    model = PeftModel.from_pretrained(model,
                                      f"best-{model_name}-finetuned-lora"
                                      )
    
    model = model.merge_and_unload()
    
    test_ds, _, image_processor = load_data(model_checkpoint,"test")
    
    classifier = pipeline(model=model,image_processor=image_processor,
                          framework="pt",device=0,batch_size=32)
    
    
    
    
    
    