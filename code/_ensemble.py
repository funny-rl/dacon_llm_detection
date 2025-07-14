import os
import sys
import torch
import argparse
import pathlib
import pandas as pd

from typing import List, Dict
from datasets import Dataset

from config.ensemble_config import get_config
from utils.utils import (
    TEST_KEY,
    PARAGRAPH_TEXT,
    compute_metrics, 
    # set_split_dataset,
    load_model_chpts,
    _load_model,
    sum_under_half,
    predict_probabilities,
    run_parallel_inference,
    run_parallel_ml
)
from utils.dataset import (
    load_test_dataset,
)


def main(all_args: argparse.Namespace) -> None:
    
    data_dir: str = all_args.data_dir
    max_length: int = all_args.max_length
    batch_size: int = all_args.batch_size
    eval_interval: int = all_args.eval_interval
    ensemble_models: Dict[str, List[str]] = all_args.ensemble_models
    num_models: int = sum([len(models) for models in ensemble_models.values()])
    model_names: List[str] = [model for models in ensemble_models.values() for model in models]
    
    test_result = pd.read_csv("./data/sample_submission.csv")
    del test_result["generated"]

    
    # 1. 기존 BERT 추론
    predictions_list = run_parallel_inference(
        model_names, 
        max_length
    )

    # 2. 추가: BERT 임베딩 기반 XGBoost 추론
    ml_predictions_list = run_parallel_ml(
        model_names, 
        max_length
    )

    # 3. 통합
    predictions_list.extend(ml_predictions_list)
    
    for idx, preds in enumerate(predictions_list):
        test_result[f"generated_{idx}"] = preds

    test_result["generated"] = test_result.apply(sum_under_half, axis=1)

    for idx in range(2*len(model_names)):
        del test_result[f"generated_{idx}"]

    test_result.to_csv(f"ensemable.csv", index=False, encoding="utf-8")
    return 


if __name__ == "__main__":
    args=sys.argv[1:]
    all_args: argparse.Namespace = get_config(args)
    main(all_args)