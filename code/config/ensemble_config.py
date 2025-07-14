import argparse

def get_config(args):
    parser = argparse.ArgumentParser(
        description='LLM-Detection', formatter_class=argparse.RawDescriptionHelpFormatter)
    function = parser.add_argument
    function(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing the dataset files (train.csv, valid.csv, test.csv)."
    )
    function(
        "--ensemble_models",
        type=dict,
        default={
            "bert_classification": [                                    # 1024 / 0.869
                "klue/roberta-large"
            ]
        },
    )
    function(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization."
    )
    function(
        "--eval_interval",
        type=int,
        default=150,
        help="Number of steps between evaluations during training."
    )
    function(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for training and evaluation per each gpu."
    )
    all_args = parser.parse_known_args(args)[0]
    return all_args