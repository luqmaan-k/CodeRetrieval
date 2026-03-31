import json
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import torch

def train_ner_model():
    print("Initializing NER Model training...")
    
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and Parse Data
    ner_path = Path("data/raw/stackoverflow_ner_train.json")
    raw_data = []
    unique_tags = set()
    
    with open(ner_path, 'r') as f:
        # Limit to 5000 for testing/setup
        for i, line in enumerate(f):
            if i >= 5000: break
            item = json.loads(line)
            raw_data.append(item)
            unique_tags.update(item['ner_tags'])
            
    labels = sorted(list(unique_tags))
    label2id = {tag: i for i, tag in enumerate(labels)}
    id2label = {i: tag for tag, i in label2id.items()}
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True, padding=False
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100) # Special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                else:
                    # Sub-words: either same label or -100 (standard is -100 to only train on first token)
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Create dataset
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.train_test_split(test_size=0.2)
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # Initialize Model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="models/ner_checkpoints",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8, # NER is more memory intensive per token
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )

    print("Starting NER training (test run)...")
    trainer.train()
    
    model_path = Path("models/ner_model_final")
    trainer.save_model(model_path)
    print(f"NER model saved to {model_path}")

if __name__ == "__main__":
    train_ner_model()
