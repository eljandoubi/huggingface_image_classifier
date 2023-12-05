"""
the main script that load the data
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
from datasets import load_dataset
from preprocessing import preprocessing


def load_data(model_checkpoint, mode):
    """load data, split it into train and validation or test set
    and return them along with the image processor"""

    # check if the mode is available
    assert mode in ["train", "test"]

    # load the processors
    preprocess_train, preprocess_val, image_processor = preprocessing(
        model_checkpoint)

    # load all images
    dataset = load_dataset("imagefolder", data_dir="images",
                           split=mode,
                           drop_labels=False)

    if mode == "train":

        # split data
        # the data is not uniformly distributed, so we have to use
        # stratify_by_column
        splits = dataset.train_test_split(
            test_size=0.1, stratify_by_column="label")

        train_ds = splits["train"]
        val_ds = splits["test"]

        # set the processor to its solit
        train_ds.set_transform(preprocess_train)
        val_ds.set_transform(preprocess_val)

    else:
        # delete the column label because it is not real
        val_ds = dataset.remove_columns("label")
        train_ds = None

    return train_ds, val_ds, image_processor
