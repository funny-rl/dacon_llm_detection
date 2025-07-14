import argparse

def get_config(args):
    parser = argparse.ArgumentParser(
        description='LLM-Detection', formatter_class=argparse.RawDescriptionHelpFormatter)
    function = parser.add_argument
    function(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization."
    )
    function(
        "--model_name", 
        type=str, 
        default="kykim/bert-kor-base",
        help="Name of the pre-trained model to use."
    )
    function(
        "--split_raio",
        default=1.0,
        type=float,
        help="Ratio of the dataset to use for training. Default is 1.0 (100%)."
    )
    function(
        "--data_dir", 
        type=str, 
        default="./data/aug10000", 
        help="Directory containing the dataset files ex. (train.csv, valid.csv, test.csv)."
    )
    function(
        "--test_data_dir", 
        type=str, 
        default="./data", 
        help="Directory containing the dataset files ex. (train.csv, valid.csv, test.csv)."
    )
    function(
        "--batch_size_per_device", 
        type=int, 
        default=32, 
        help="Batch size for training and evaluation per device."
    )
    all_args = parser.parse_known_args(args)[0]
    return all_args