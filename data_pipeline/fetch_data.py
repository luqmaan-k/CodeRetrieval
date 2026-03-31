import os
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Config
DATA_DIR = Path("data/raw")

def download_datasets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. CoSQA
    print("Fetching CoSQA...")
    try:
        # Using a reliable version found in search
        cosqa = load_dataset("gonglinyuan/CoSQA")
        cosqa["train"].to_json(DATA_DIR / "cosqa_train.json")
        print("CoSQA saved.")
    except Exception as e:
        print(f"Error fetching CoSQA: {e}")

    # 2. CoNaLa
    print("Fetching CoNaLa...")
    try:
        # Switching to codeparrot/conala-mined-curated as it is Parquet-based
        # and avoids the 'dataset scripts are no longer supported' error in neulab/conala
        conala = load_dataset("codeparrot/conala-mined-curated")
        # This dataset has 'curated' and 'mined' as splits/subsets sometimes, 
        # but let's check what we get. Usually it has 'train'.
        if "train" in conala:
            conala["train"].to_json(DATA_DIR / "conala_train.json")
        else:
            # If it's a DatasetDict with other keys, just save the first one or curated if exists
            split_to_save = "curated" if "curated" in conala else list(conala.keys())[0]
            conala[split_to_save].to_json(DATA_DIR / "conala_train.json")
        print("CoNaLa saved.")
    except Exception as e:
        print(f"Error fetching CoNaLa: {e}")

    # 3. StackOverflow NER
    print("Fetching StackOverflow NER...")
    try:
        ner = load_dataset("mrm8488/stackoverflow-ner")
        ner["train"].to_json(DATA_DIR / "stackoverflow_ner_train.json")
        print("StackOverflow NER saved.")
    except Exception as e:
        print(f"Error fetching NER: {e}")

if __name__ == "__main__":
    download_datasets()
