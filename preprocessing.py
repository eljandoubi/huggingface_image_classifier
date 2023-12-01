"""
The image preprocessor script
Author: Abdelkarim eljandoubi
date: Nov 2023
"""

from transformers import AutoImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import torch



def preprocessing(model_checkpoint:str)->tuple[callable]:
    """
    From the model checkpoint, it returns two callable that trait train and validation datasets
    """
    
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )


    def preprocess_train(example_batch:dict)->dict:
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch


    def preprocess_val(example_batch:dict)->dict:
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    
    return preprocess_train, preprocess_val, image_processor


def collate_fn(examples:dict)->dict:
    """Data Collactor"""
    
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    
    return {"pixel_values": pixel_values, 'labels': labels}

