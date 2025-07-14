import argparse

def get_config(args):
    parser = argparse.ArgumentParser(
        description='LLM-Detection', formatter_class=argparse.RawDescriptionHelpFormatter)
    function = parser.add_argument
    function(
        "--data_dir", 
        type=str, 
        default="./data", 
        help="Directory containing the dataset files ex. (train.csv, valid.csv, test.csv)."
    )
    function(
        "--model_name", 
        type=str, 
        default="kykim/bert-kor-base",
        choices=[
            "kykim/electra-kor-base",                                           
            "klue/roberta-large", 
            "kykim/funnel-kor-base",
            "beomi/KcELECTRA-base",        
        ], 
        help="Name of the pre-trained model to use."
    )
    function(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization."
    )
    all_args = parser.parse_known_args(args)[0]
    return all_args