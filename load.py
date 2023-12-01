"""
the main script that load the data
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
from datasets import load_dataset
from preprocessing import preprocessing

def load_data(model_checkpoint,mode):

    """load data"""
    
    assert mode in ["train","test"]
    
    preprocess_train, preprocess_val, image_processor = preprocessing(model_checkpoint)

    dataset = load_dataset("imagefolder", data_dir="images",
                           split=mode,
                           drop_labels=False)
    
    if mode=="train":
        # the data is not uniformly distributed, so we have to use stratify_by_column
        
        splits = dataset.train_test_split(test_size=0.1,stratify_by_column="label")
        
        train_ds = splits["train"]
        val_ds = splits["test"]
        
        train_ds.set_transform(preprocess_train)
        val_ds.set_transform(preprocess_val)
            
    else:
        val_ds = dataset.remove_columns("label")
        train_ds=None
            
    
    return train_ds,val_ds,image_processor