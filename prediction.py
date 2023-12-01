"""
the main script that predicts labels
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
from transformers import AutoModelForImageClassification, ImageClassificationPipeline
from peft import PeftModel
import numpy as np
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
    
    classifier = ImageClassificationPipeline(model=model,
                                             image_processor=image_processor,
                                             framework="pt",
                                             device=0,
                                             batch_size=32)
    
    results = classifier(test_ds["image"],top_k=1)
    
    
    results = map(lambda x:int(x[0]['label'][-1]),results)
    
    results = np.array(list(results))
    
    np.savetxt("predictions.txt",results,fmt="%d")
    
    
    
    
    