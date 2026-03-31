import os
import sys
import pandas as pd
import torch
import time
import json
import logging
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# Setup detailed logging
LOG_FILE = "models/logs/intent_training.log"
os.makedirs("models/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LoggerWriter:
    def __init__(self, level):
        self.level = level
    def write(self, message):
        if message.strip() != "":
            self.level(message.strip())
    def flush(self):
        pass
    def isatty(self):
        return False
    def fileno(self):
        return 1 # Standard stdout

# Redirect stdout and stderr to the logger
sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

def train_intent_classifier():
    logger.info("Initializing Intent Classifier training (100k subset)...")
    start_time = time.time()
    
    try:
        # Load mapped data
        data_path = Path("data/processed/mapped_intents.csv")
        df = pd.read_csv(data_path)
        df = df[df['intent'] != 'Explain_Code']
        
        # Label encoding
        labels = sorted(df['intent'].unique())
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in label2id.items()}
        df = df.assign(labels=df['intent'].map(label2id))
        
        # Take a balanced subset (approx 100k total)
        subset_df = df.groupby('labels', group_keys=False).apply(lambda x: x.sample(min(len(x), 33333), random_state=42)).reset_index(drop=True)
        subset_df = subset_df.assign(labels=subset_df['intent'].map(label2id))
        
        logger.info(f"Training on balanced subset of {len(subset_df)} samples.")

        # Model and Tokenizer
        model_name = "microsoft/graphcodebert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def tokenize_function(examples):
            result = tokenizer(examples["query"], truncation=True, padding=True, max_length=128)
            result["labels"] = [int(l) for l in examples["labels"]]
            return result

        # Create dataset
        dataset = Dataset.from_dict({
            "query": subset_df["query"].astype(str).values.tolist(),
            "labels": subset_df["labels"].values.tolist()
        })
        dataset = dataset.train_test_split(test_size=0.1)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Initialize Model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label
        )

        # Training Arguments
        training_args = TrainingArguments(
            output_dir="models/intent_classifier_checkpoints",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
            logging_dir="models/logs/tensorboard"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        logger.info("Starting training...")
        train_result = trainer.train()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Save final model
        model_path = Path("models/intent_classifier_final")
        trainer.save_model(model_path)
        
        # Log stats
        stats = {
            "model_type": "Intent Classifier (GraphCodeBERT)",
            "samples": len(subset_df),
            "epochs": 3,
            "duration_seconds": duration,
            "final_loss": train_result.training_loss,
            "metrics": trainer.evaluate()
        }
        
        logger.info(f"Intent model saved to {model_path}. Time: {duration/60:.2f} minutes.")
        return stats

    except Exception as e:
        logger.exception("An error occurred during training:")
        raise

if __name__ == "__main__":
    try:
        stats = train_intent_classifier()
        os.makedirs("models", exist_ok=True)
        with open("models/intent_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        sys.exit(1)
