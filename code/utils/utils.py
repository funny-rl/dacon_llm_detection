import re
import json
import torch
import evaluate
import multiprocessing as mp
import pathlib
import numpy as np
import joblib
import xgboost as xgb

from typing import List
from tqdm import tqdm

from datasets import Dataset, load_dataset, ClassLabel, DatasetDict, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils.dataset import (
    TRAIN_KEY,
    VALID_KEY,
    TEST_KEY,
    PARAGRAPH_TEXT,
    load_test_dataset
)

TRAIN_KEY = "train"
VALID_KEY = "valid"
TEST_KEY = "test"
PARAGRAPH_TEXT = "paragraph_text"

def compute_metrics(pred):
    roc_auc = evaluate.load("roc_auc")
    logits, labels = pred
    probs = logits[:, 1]  
    auc = roc_auc.compute(prediction_scores=probs, references=labels)["roc_auc"]  
    return {"auroc": auc}

def load_model_chpts(ROOT_DIR, BEST_JSON):
    if BEST_JSON.exists():
        with open(BEST_JSON) as f:
            state = json.load(f)
        best_ckpt = state["best_model_checkpoint"]
    else:
        ckpts = sorted(ROOT_DIR.glob("checkpoint-*"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
        best_ckpt = str(ckpts[-1])
    print(f"✨ Best checkpoint detected: {best_ckpt} ✨")
    return best_ckpt

def predict_probabilities(predictions):
    generated_column: List[int] = []
    for res in tqdm(predictions, total=len(predictions), desc="Processing Inference Results"):
        llm_prob = next((item["score"] for item in res if item["label"] == 1), 0.0) 
        generated_column.append(llm_prob)
        
    return generated_column

def _load_model(best_ckpt):
    model = AutoModelForSequenceClassification.from_pretrained(
        best_ckpt,
        torch_dtype="auto",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if num_parameters >= 1_000_000_000:
        num_in_billion = num_parameters / 1_000_000_000
        print(f"✨ Model has {num_parameters:,} trainable parameters ({num_in_billion:.2f}B) ✨")
    else:
        num_in_million = num_parameters / 1_000_000
        print(f"✨ Model has {num_parameters:,} trainable parameters ({num_in_million:.2f}M) ✨")
    
    return model


def num_under_half(row, n_ensemble_models):
    under_half = [
        row[f"generated_{idx}"]
        for idx in range(n_ensemble_models)
        if row[f"generated_{idx}"] < 0.5
    ]
    over_half = [
        row[f"generated_{idx}"]
        for idx in range(n_ensemble_models)
        if row[f"generated_{idx}"] >= 0.5
    ]
    return under_half, over_half

def sum_under_half(row):
    """
    """
    n_ensemble_models = sum("generated" in col for col in row.index)
    
    under_half, over_half = num_under_half(row, n_ensemble_models)
    
    # if n_ensemble_models > len(under_half) >= 1:
    #     print(f"Under half: {len(under_half)}, Over half: {len(over_half)}")

    # if len(under_half) == n_ensemble_models / 2:
    #     return  sum(over_half) / len(over_half) if sum(over_half) / len(over_half) >  (1 - sum(under_half) / len(under_half)) else sum(under_half) / len(under_half)
    # elif len(under_half) > n_ensemble_models / 2:
    #     return sum(under_half) / len(under_half)
    # else:
    #     return sum(over_half) / len(over_half)
    
    if len(under_half) > n_ensemble_models / 2:
        return 0 if len(under_half) == n_ensemble_models else min(under_half)
    
    elif len(over_half) > n_ensemble_models / 2:
        return 1 if len(over_half) == n_ensemble_models else max(over_half)
    
    else:
        if sum(over_half) / len(over_half) >  (1 - sum(under_half) / len(under_half)):
            return max(over_half)
        else:
            return min(under_half)
        
def run_inference_worker(model_name: str, model_idx: int, device_id: int, max_length: int, queue):
    print(f"🚀 Running model {model_name} on GPU {device_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    test_dataset = load_test_dataset(
        data_dir="./data"
    )
    def test_tokenize(example):
        return tokenizer(example[PARAGRAPH_TEXT], padding="max_length", truncation=True, max_length=max_length)
    test_dataset = test_dataset.map(test_tokenize, batched=True)
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_texts = test_dataset[TEST_KEY][PARAGRAPH_TEXT]  # 보통 "paragraph_text" 컬럼
    
    ROOT_DIR = pathlib.Path(f"./ann_models/{model_name}")
    BEST_JSON = ROOT_DIR / "trainer_state.json"
    best_ckpt = load_model_chpts(ROOT_DIR, BEST_JSON)
    print(best_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        best_ckpt,
        torch_dtype="auto",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=device_id,
        function_to_apply="sigmoid",
        return_all_scores=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        batch_size=32,
    )

    predictions = pipe(test_texts)
    generated_column = predict_probabilities(predictions)

    queue.put((model_idx, generated_column))  # 순서 유지를 위해 인덱스도 같이


def run_parallel_inference(model_names, max_length):
    num_gpus = torch.cuda.device_count()
    manager = mp.Manager()
    queue = manager.Queue()
    processes = []

    for idx, model_name in enumerate(model_names):
        device_id = idx % num_gpus
        p = mp.Process(target=run_inference_worker,
                       args=(model_name, idx, device_id, max_length, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        
    # 예측 결과 수집
    results = [None] * len(model_names)
    while not queue.empty():
        idx, preds = queue.get()
        results[idx] = preds

    return results

def run_ml_worker(bert_model_name: str, device_id: int, test_embeddings: np.ndarray, queue, model_idx: int):
    """
    병렬로 XGBoost 모델 예측 수행
    """
    bert_model_name = bert_model_name.split("/")[-1]
    model_dir = pathlib.Path(f"./xgb/models/{bert_model_name}.json")
    
    assert model_dir.exists(), f"{model_dir} not found"
    
    clf = xgb.Booster()
    clf.load_model(str(model_dir))
    
    dtest = xgb.DMatrix(test_embeddings)
    
    preds = clf.predict(dtest)
    
    queue.put((model_idx, preds))


def run_parallel_ml(model_names, max_length):
    """
    model_names: List[str] = ["kykim", "klue/roberta-large"]
    """
    print("🚀 Running ML (XGBoost) inference in parallel...")

    num_cpus = mp.cpu_count()
    queue = mp.Manager().Queue()
    processes = []

    results = [None] * len(model_names)
    
    for idx, model_name in enumerate(model_names):
        # 1️⃣ 임베딩 로드
        model_name = model_name.split("/")[-1]
        # print(model_name)
        emb_path = pathlib.Path(f"./xgb/data/{model_name}.npz")
        assert emb_path.exists(), f"{emb_path} not found"
        # data = np.load(emb_path)
        # print("✅ npz 내부 키 목록:", data.files)
        test_emb = np.load(emb_path)["X_test"]

        # 2️⃣ 프로세스 생성
        p = mp.Process(
            target=run_ml_worker,
            args=(model_name, idx % num_cpus, test_emb, queue, idx)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    while not queue.empty():
        idx, preds = queue.get()
        results[idx] = preds
    #print(results)
    return results 