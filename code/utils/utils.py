import re
import json
import torch
import evaluate

from typing import List
from tqdm import tqdm

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