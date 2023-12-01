"""
the main script that load the data
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
from datasets import load_dataset
from preprocessing import preprocessing

def load_data(model_checkpoint:str,mode:str):

    """load data"""
    
    assert mode in ["search","train","test"]
    
    preprocess_train, preprocess_val, image_processor = preprocessing(model_checkpoint)

    dataset = load_dataset("imagefolder", data_dir="images",
                           split="test" if mode=="test" else "train",
                           drop_labels=False)
    if mode=="search":
        # the data is not uniformly distributed, so we have to use stratify_by_column
        splits = dataset.train_test_split(test_size=0.2,stratify_by_column="label")
        
        train_ds = splits["train"]
        val_ds = splits["test"]
        
        train_ds.set_transform(preprocess_train)
        val_ds.set_transform(preprocess_val)
        
    else:
        train_ds = dataset[mode]
        
        if mode=="train":
            train_ds.set_transform(preprocess_train)
            
        else:
            train_ds = train_ds.remove_columns("label")
            
            
        val_ds=None
    
    return train_ds,val_ds,image_processor