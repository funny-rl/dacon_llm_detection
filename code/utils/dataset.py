import torch

from tqdm import tqdm
from datasets import load_dataset, ClassLabel, DatasetDict, concatenate_datasets

TRAIN_KEY = "train"
VALID_KEY = "valid"
TEST_KEY = "test"
PARAGRAPH_TEXT = "paragraph_text"

def split_dataset(data_dir: str, ratio: float = 1.0, seed: int = 42):
    dataset: DatasetDict = load_dataset(
        "csv",
        data_files={
            TRAIN_KEY: f"{data_dir}/train.csv",
            VALID_KEY: f"{data_dir}/valid.csv",
        }
    )
    label_class = ClassLabel(names=["0", "1"], num_classes=2)
    dataset = dataset.cast_column("label", label_class)

    full_train = dataset[TRAIN_KEY]

    label_0 = full_train.filter(lambda ex: ex["label"] == 0)
    label_1 = full_train.filter(lambda ex: ex["label"] == 1)

    n0 = int(len(label_0) * ratio)
    n1 = int(len(label_1) * ratio)
    sampled_0 = label_0.shuffle(seed=42).select(range(n0))
    sampled_1 = label_1.shuffle(seed=42).select(range(n1))

    concated_dataset = [sampled_0, sampled_1]
    balanced_sampled = concatenate_datasets(concated_dataset)
    balanced_sampled = balanced_sampled.shuffle(seed=seed)

    dataset[TRAIN_KEY] = balanced_sampled

    return dataset

def load_test_dataset(data_dir: str):
    test_dataset =  load_dataset(
        "csv",
        data_files={TEST_KEY: f"{data_dir}/test.csv"}
    )

    return test_dataset

def get_embeddings(dataset, model, tokenizer, max_length: int, batch_size: int = 32, text_column: str = "text"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_embeddings = []
    texts = dataset[text_column]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="✨ Extracting Embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts, 
                padding="max_length", 
                truncation=True, 
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()