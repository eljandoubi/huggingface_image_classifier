"""
the main script that predicts labels
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
import os
from transformers import (AutoModelForImageClassification,
                          ImageClassificationPipeline,
                          AutoConfig)

from peft import PeftModel
import numpy as np
from load import load_data


def predict(model_checkpoint):
    """Get prediction of the test data set"""

    # check if we already have predictions
    if os.path.isfile("predictions.txt"):
        return

    # get the model configuration and set the number of labels to 2 (0-1)
    config = AutoConfig.from_pretrained(model_checkpoint, num_labels=2)

    # load the model
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # get the model name
    model_name = model_checkpoint.split("/")[-1]

    # load lora layers
    model = PeftModel.from_pretrained(model,
                                      f"best-{model_name}-finetuned-lora"
                                      )
    # merge the two for faster inference
    model = model.merge_and_unload()

    # load test data
    _, test_ds, image_processor = load_data(model_checkpoint, "test")

    # define the inference pipeline
    classifier = ImageClassificationPipeline(model=model,
                                             image_processor=image_processor,
                                             framework="pt",
                                             device=0,
                                             batch_size=32)

    # get predictions
    results = classifier(test_ds["image"], top_k=1)

    # transfrom predictions to 0-1
    results = map(lambda x: int(x[0]['label'].split("_")[-1]), results)

    # save to np array
    results = np.array(list(results))

    # save it to predictions.txt file
    np.savetxt("predictions.txt", results, fmt="%d")
