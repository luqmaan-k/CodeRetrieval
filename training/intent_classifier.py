import os
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

def train_intent_classifier():
    print("Initializing Intent Classifier training...")
    
    # Load mapped data
    data_path = Path("data/processed/mapped_intents.csv")
    df = pd.read_csv(data_path)
    
    # Drop minority class and clean
    df = df[df['intent'] != 'Explain_Code']
    
    # Label encoding
    labels = sorted(df['intent'].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df = df.assign(labels=df['intent'].map(label2id))
    
    # Take a balanced subset
    subset_df = df.groupby('labels', group_keys=False).apply(lambda x: x.sample(min(len(x), 5000), random_state=42)).reset_index(drop=True)
    subset_df = subset_df.assign(labels=subset_df['intent'].map(label2id))
    print(f"Subset columns: {subset_df.columns}")
    
    # Model and Tokenizer
    model_name = "microsoft/graphcodebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        result = tokenizer(examples["query"], truncation=True, padding=True, max_length=128)
        result["labels"] = examples["labels"]
        return result

    # Create dataset
    dataset = Dataset.from_dict({
        "query": subset_df["query"].values.tolist(),
        "labels": subset_df["labels"].values.tolist()
    })
    dataset = dataset.train_test_split(test_size=0.2)
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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1, 
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Starting training (test run)...")
    trainer.train()
    
    # Save final model
    model_path = Path("models/intent_classifier_final")
    trainer.save_model(model_path)
    print(f"Intent model saved to {model_path}")

if __name__ == "__main__":
    train_intent_classifier()
