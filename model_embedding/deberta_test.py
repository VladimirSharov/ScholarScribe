from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import evaluate
import json

# Specify paths to each split
train_path = "data_split/full_dataset_train.json"
val_path = "data_split/full_dataset_val.json"
test_path = "data_split/full_dataset_test.json"

# Load datasets directly
train_data = Dataset.from_pandas(pd.read_json(train_path))
val_data = Dataset.from_pandas(pd.read_json(val_path))
test_data = Dataset.from_pandas(pd.read_json(test_path))

# Combine into a DatasetDict
dataset = DatasetDict({
    'train': train_data,
    'validation': val_data,
    'test': test_data
})

# Combine tags from 'additional_tags' and 'subject_tags' into 'all_tags'
def combine_tags(example):
    all_tags = list(set(example["additional_tags"] + example["subject_tags"]))  # Deduplicate tags
    example["all_tags"] = all_tags
    return example

dataset = dataset.map(combine_tags)

# Define unique tags and create mappings
unique_tags = sorted({tag for item in dataset["train"] for tag in item["all_tags"]})
tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
id2tag = {idx: tag for tag, idx in tag2id.items()}

# Initialize the tokenizer
model_name = 'FacebookAI/xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function to tokenize and encode labels
def preprocess_function(examples):
    # Combine title and abstract for input text
    texts = [f"{title}. {abstract}" for title, abstract in zip(examples["thesis_title"], examples["abstract"])]
    
    # Encode tags to multi-hot vectors
    labels = [[tag2id[tag] for tag in tags if tag in tag2id] for tags in examples["all_tags"]]
    multi_hot_labels = np.zeros((len(labels), len(tag2id)), dtype=float)
    for i, label_list in enumerate(labels):
        for label in label_list:
            multi_hot_labels[i][label] = 1.0

    # Tokenize text fields
    tokenized_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    tokenized_inputs["label"] = multi_hot_labels.tolist()
    return tokenized_inputs

# Tokenize and encode the dataset for all splits
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Data collator with padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define evaluation metrics
metric = evaluate.combine(["f1", "precision", "recall"])

# Compute metrics function
def compute_metrics(p):
    pred_labels = np.array(p.predictions) > 0.5
    return metric.compute(predictions=pred_labels, references=p.label_ids)

# Initialize the model for multi-label classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(tag2id),
    problem_type="multi_label_classification"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="output_dir",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
