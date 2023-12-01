"""
the main script
Author: Abdelkarim eljandoubi
date: Nov 2023
"""

import logging
import argparse
from extract import extract
from hyper_params_search import search
from train import train_best
from prediction import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def main(args):
    
    logger.info('Start extraction')
    extract()
    
    logger.info('Start hyperparameter search')
    search(args.model_checkpoint, args.n_trials)
    
    logger.info('Start train the model with the best found hyperparameters')
    train_best(args.model_checkpoint)
    
    logger.info('Start get predictions')
    predict(args.model_checkpoint)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This is a the lora fine-tuner"
    )
    
    parser.add_argument("--model_checkpoint", type=str,
                        help="the path to the hugging face moadel",
                        default="google/vit-base-patch16-224-in21k",
                        )
    
    parser.add_argument("--n_trials", type=int,
                        help="the number of trials to look for optimal parametes",
                        default=1
                        )
    
    argms = parser.parse_args()

    main(argms)