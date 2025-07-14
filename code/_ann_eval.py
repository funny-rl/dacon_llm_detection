import sys
import time
import torch 
import pathlib
import datetime
import argparse
import pandas as pd

from typing import List

from utils.dataset import (
    TEST_KEY,
    PARAGRAPH_TEXT,
    load_test_dataset
)
from utils.utils import (
    load_model_chpts,
    predict_probabilities,
)

from config.ann_eval_config import get_config

from transformers import (
    AutoTokenizer,
    pipeline,
    AutoModelForSequenceClassification,
)

def main(all_args: argparse.Namespace) -> None:
    max_length: int = all_args.max_length
    model_name: str = all_args.model_name
    
    
    test_dataset = load_test_dataset(
        data_dir=all_args.data_dir
    )
    test_texts = test_dataset[TEST_KEY][PARAGRAPH_TEXT]
    submission_format = pd.read_csv("./data/sample_submission.csv")

    output_dir: str = f"./ann_models/{model_name}"
    ROOT_DIR: pathlib.Path = pathlib.Path(output_dir)
    BEST_JSON: pathlib.Path = ROOT_DIR / "trainer_state.json"
    best_ckpt = load_model_chpts(ROOT_DIR, BEST_JSON)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        best_ckpt,
        torch_dtype="auto",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    cls = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,  
        device=0 if torch.cuda.is_available() else -1,
        function_to_apply="sigmoid",
        return_all_scores=True,
        padding=True,
        truncation=True, 
        max_length=max_length,
    )
    
    predictions = cls(test_texts)
    generated_column: List[int] = predict_probabilities(predictions)

    submission_format["generated"] = generated_column
    submission_format.to_csv(f"{model_name.split('/')[-1]}_{datetime.datetime.now().strftime('%Y년%m월%d일 %H시:%M분:%S초')}.csv", index=False, encoding="utf-8")
    return 

if __name__ == "__main__":
    args = sys.argv[1:]
    main(get_config(args))