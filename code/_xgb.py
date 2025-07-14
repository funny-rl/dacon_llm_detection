import sys
import pathlib
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score

from utils.dataset import (
    TRAIN_KEY,
    VALID_KEY,
    TEST_KEY,
    PARAGRAPH_TEXT,
    get_embeddings,
    split_dataset,
    load_test_dataset
)
from utils.utils import (
    load_model_chpts,
)

from config.xgb_config import get_config

from transformers import AutoModel, AutoTokenizer

def main(all_args: argparse.Namespace) -> None:
    max_length = all_args.max_length
    batch_size_per_device: int = all_args.batch_size_per_device
   
    model_name: str = all_args.model_name
    
    output_dir: str = f"./ann_models/{model_name}"
    ROOT_DIR: pathlib.Path = pathlib.Path(output_dir)
    BEST_JSON: pathlib.Path = ROOT_DIR / "trainer_state.json"
    best_ckpt = load_model_chpts(ROOT_DIR, BEST_JSON)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(best_ckpt)
    
    embedding_dir = pathlib.Path("./xgb/data")
    safe_model_name = pathlib.Path(model_name).name
    embedding_filename = embedding_dir / f"{safe_model_name}_embeddings.npz"

    if embedding_filename.exists():
        print(f"✅ Found pre-computed embeddings. Loading from {embedding_filename}...")
        data = np.load(embedding_filename)
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        X_test = data['X_test']
    
    else:
        dataset = split_dataset(
            data_dir=all_args.data_dir,
        )
        test_dataset = load_test_dataset(
            data_dir=all_args.test_data_dir,
        )

        def tokenize(example):
            return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)
        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        def test_tokenize(example):
            return tokenizer(example[PARAGRAPH_TEXT], padding="max_length", truncation=True, max_length=max_length)
        test_dataset = test_dataset.map(test_tokenize, batched=True)
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        X_train = get_embeddings(dataset[TRAIN_KEY], model, tokenizer, max_length, batch_size_per_device)
        y_train = np.array(dataset[TRAIN_KEY]['label'])
        
        X_valid = get_embeddings(dataset[VALID_KEY], model, tokenizer, max_length, batch_size_per_device)
        y_valid = np.array(dataset[VALID_KEY]['label'])
        
        X_test = get_embeddings(test_dataset[TEST_KEY], model, tokenizer, max_length, batch_size_per_device, text_column=PARAGRAPH_TEXT)
        
        print(f"\n💾 Saving embeddings to {embedding_filename}...")
        np.savez_compressed(
            embedding_filename,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test
        )
        print("✅ Embeddings saved successfully!")

    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,    
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='auc',  
        early_stopping_rounds=100,
        random_state=42,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100
    )
    
    valid_preds = xgb_model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, valid_preds)
    print(f"\nValidation ROC AUC Score: {auc_score:.4f}")
    
    model_save_path = f"./xgb/models//{safe_model_name}_{auc_score:.4f}.json"
    xgb_model.save_model(model_save_path)
    
    # 5. 테스트 데이터로 예측 및 제출 파일 생성
    print("\nPredicting on test data and creating submission file...")
    test_preds = xgb_model.predict_proba(X_test)[:, 1]
    
    submission = pd.read_csv("./data/sample_submission.csv") # 샘플 제출 파일 경로
    submission['generated'] = test_preds
    submission.to_csv(f"xgboost_submission_{model_name.replace('/', '_')}.csv", index=False)

    return 

if __name__ == "__main__":
    args=sys.argv[1:]
    main(get_config(args))