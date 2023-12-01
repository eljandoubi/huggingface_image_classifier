"""
the main script that initialize the trainer
Author: Abdelkarim eljandoubi
date: Nov 2023
"""
import torch
from transformers import TrainingArguments, Trainer
from preprocessing import collate_fn
from model import model_from_checkpoint
from metrics import calculate_hter
from load import load_data


def lora_trainer(model_checkpoint,
                 tf32=True,
                 learning_rate=1e-4,
                 per_device_train_batch_size=8,
                 gradient_accumulation_steps=4,
                 num_train_epochs=10,
                 r = 16,
                 lora_alpha = 16,
                 lora_dropout = 0.1,
                 bias = "none",
                 mode="search"
                 ):
    
    assert mode in ["search","train"]
    
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    
    model_name = model_checkpoint.split("/")[-1]
    
    args = TrainingArguments(
        f"{model_name}-finetuned-lora",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        tf32=tf32,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="hter",
        push_to_hub=False,
        label_names=['labels'],
        greater_is_better=False,
    )
    
    train_ds, val_ds, image_processor = load_data(model_checkpoint,mode)
    
    model_init = model_from_checkpoint(model_checkpoint,
                                       r,
                                       lora_alpha,
                                       lora_dropout,
                                       bias
                                       )
    
    trainer = Trainer(
        args=args,
        model_init=model_init,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        compute_metrics=calculate_hter,
        data_collator=collate_fn,
        )
    
    return trainer