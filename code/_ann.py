import sys
import argparse

from typing import List
from config.ann_config import get_config

from utils.utils import compute_metrics

from utils.dataset import (
    TRAIN_KEY,
    VALID_KEY,
    split_dataset
)

from transformers import (
    Trainer,
    AutoTokenizer, 
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification,
)

def main(all_args: argparse.Namespace) -> None:
    
    lr: float = all_args.lr
    split_raio: float = all_args.split_raio 
    
    max_length: int = all_args.max_length
    batch_size_per_device: int = all_args.batch_size_per_device
    valid_interval: int = all_args.valid_interval
    early_stopping_patience: int = all_args.early_stopping_patience
    
    data_dir: str = all_args.data_dir
    model_name: str = all_args.model_name
    output_dir: str = f"./ann_models/{model_name}"
    
    assert 1.0 >= split_raio >= 0.0, "split_raio must be between 0.0 and 1.0"
    
    dataset = split_dataset(
        data_dir = data_dir,
        ratio =split_raio,
        seed =42
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=valid_interval,  
        save_steps=valid_interval,
        save_strategy="steps",
        per_device_train_batch_size=batch_size_per_device,
        per_device_eval_batch_size=batch_size_per_device,
        num_train_epochs=5,
        save_total_limit=1,
        learning_rate=lr,
        lr_scheduler_type="constant",
        warmup_steps=valid_interval,
        load_best_model_at_end=True,
        metric_for_best_model="eval_auroc",
        ddp_find_unused_parameters=True,
        fp16=True, 
        logging_steps=50,
        report_to="wandb",
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: 0, 1: 1},
        label2id={0: 0, 1: 1},
    )

    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset[TRAIN_KEY],
        eval_dataset=dataset[VALID_KEY],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    trainer.train()

if __name__ == "__main__":
    args: List[str] = sys.argv[1:]
    main(get_config(args))
    