import sys
import pathlib
import argparse

import numpy as np

from config.xgb_config import (
    load_model_chpts, 
)

from transformers import AutoModel, AutoTokenizer

def main(all_args: argparse.Namespace) -> None:
    max_lent = all_args.max_length
    batch_size: int = all_args.batch_size
   
    model_dir: str = all_args.model_dir
    model_name: str = all_args.model_name
    
    output_dir: str = f"./xgb_models/{model_name}"
    ROOT_DIR: pathlib.Path = pathlib.Path(model_dir)
    BEST_JSON: pathlib.Path = ROOT_DIR / "trainer_state.json"
    best_ckpt = load_model_chpts(ROOT_DIR, BEST_JSON)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(best_ckpt)
    
    embedding_dir = pathlib.Path("./xgb_data")
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
        
        
        
    return 

if __name__ == "__main__":
    args=sys.argv[1:]
    main(get_config(args))