import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import evaluate

# Load the dataset with custom code trusted
dataset = load_dataset('knowledgator/events_classification_biotech', trust_remote_code=True)

# Inspect column names and feature types for verification
print("Column names:", dataset['train'].column_names)
print("Feature details:", dataset['train'].features)

# Confirm the 'label' field and define classes accordingly
if 'label' in dataset['train'].features:
    classes = dataset['train'].features['label'].names
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}
else:
    raise ValueError("Dataset is missing the required 'label' field.")

# Initialize tokenizer
model_path = 'microsoft/deberta-v3-small'
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Preprocessing function with dynamic key checks
def preprocess_function(example):
    text = ""
    if 'title' in example:
        text += f"{example['title']}. "
    if 'content' in example:
        text += f"{example['content']}"

    if 'all_labels' in example:
        all_labels = example['all_labels'].split(', ')
    else:
        raise ValueError("Missing 'all_labels' field in dataset example.")

    labels = [0. for _ in range(len(classes))]
    for label in all_labels:
        label_id = class2id.get(label)
        if label_id is not None:
            labels[label_id] = 1.

    example = tokenizer(text, truncation=True)
    example['labels'] = labels
    return example

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function)

# Prepare data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluation metrics
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Compute metrics for multi-label
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    labels = labels.astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions, references=labels)

# Load DeBERTa model
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(classes),
    id2label=id2class,
    label2id=class2id,
    problem_type="multi_label_classification"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
