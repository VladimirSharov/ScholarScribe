import json
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
import wandb
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

def load_data(data_dir):
    """Load train, validation, and test data - handles both JSONL and JSON array formats"""
    train_path = os.path.join(data_dir, 'train.json')
    val_path = os.path.join(data_dir, 'val.json')
    test_path = os.path.join(data_dir, 'test.json')
    
    def load_json_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            return []
            
        # Try JSONL format first (each line is a JSON object)
        try:
            lines = content.split('\n')
            data = []
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
            return data
        except json.JSONDecodeError:
            pass
        
        # Try regular JSON array format
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error loading {file_path}: {e}")
            print(f"First 200 characters of file: {content[:200]}")
            raise
    
    train_data = load_json_file(train_path)
    val_data = load_json_file(val_path)
    test_data = load_json_file(test_path)
    
    return train_data, val_data, test_data

def prepare_labels(train_data, val_data, test_data):
    """Prepare multi-label encoding"""
    all_tags = set()
    for data in [train_data, val_data, test_data]:
        for item in data:
            all_tags.update(item['tags'])
    
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_tags)])
    
    train_labels = mlb.transform([item['tags'] for item in train_data])
    val_labels = mlb.transform([item['tags'] for item in val_data])
    test_labels = mlb.transform([item['tags'] for item in test_data])
    
    return mlb, train_labels, val_labels, test_labels

# Frequency bands matching split_report.json's label_freq_bands convention,
# computed here over the training set only.
FREQ_BINS = [(3, 4), (5, 9), (10, 49), (50, 199), (200, None)]
FREQ_BIN_NAMES = ["3-4", "5-9", "10-49", "50-199", "200+"]

# Old global sweep [0.1..0.7] never fired for this label space: sigmoid outputs
# stay below 0.1 almost everywhere given ~5 positives out of 22,027 classes/example.
# Kept coarse (10 points) since find_best_threshold_per_group runs every eval and
# each grid point costs a full (n_samples x n_labels) pass.
THRESHOLD_GRID = np.round(np.arange(0.01, 0.51, 0.05), 2)


def compute_tag_bucket_ids(train_labels):
    """Assign each label column to a frequency bucket, aligned to mlb.classes_ order."""
    freqs = train_labels.sum(axis=0)
    bucket_ids = np.zeros(len(freqs), dtype=int)
    for bucket_id, (lo, hi) in enumerate(FREQ_BINS):
        mask = (freqs >= lo) if hi is None else (freqs >= lo) & (freqs <= hi)
        bucket_ids[mask] = bucket_id
    return bucket_ids


def compute_pos_weight(train_labels):
    """Per-class pos_weight for BCEWithLogitsLoss, log-scaled so rare tags (raw
    neg/pos ratio in the tens of thousands) don't destabilize training.

    sqrt-damped on top of log1p: the 2026-07-12 run used raw log1p(neg/pos)
    (mean ~9.3x across the board, since even the "common" end of a heavily
    imbalanced label space is still rare in absolute terms) and it pushed the
    model into severe over-prediction (peaked near-universal positive
    predictions at step ~1000) that only partially self-corrected within the
    run's budget. sqrt() keeps the same rank ordering (rarer tags still get
    more weight) but compresses the range (this dataset: raw mean 8.28, max
    9.47 -> sqrt mean 2.87, max 3.08), aiming for enough gradient signal on
    rare tags without repeating the overshoot."""
    pos = np.maximum(train_labels.sum(axis=0).astype(np.float64), 1)
    neg = train_labels.shape[0] - pos
    return torch.tensor(np.sqrt(np.log1p(neg / pos)), dtype=torch.float32)


def find_best_threshold_per_group(probs, labels, bucket_ids, threshold_grid):
    """Per frequency bucket, find the threshold maximizing that bucket's own F1-micro.

    Computes TP/FP/FN via plain numpy boolean ops rather than calling sklearn's
    f1_score per (bucket, threshold) pair - with 22,027 columns and a multi-point
    grid, the sklearn-call version took minutes per eval; this takes seconds."""
    labels_bool = labels.astype(bool)
    unique_buckets = np.unique(bucket_ids)
    cols_by_bucket = {b: np.where(bucket_ids == b)[0] for b in unique_buckets}
    labels_by_bucket = {b: labels_bool[:, cols] for b, cols in cols_by_bucket.items()}

    best_f1 = {b: 0.0 for b in unique_buckets}
    threshold_map = {b: threshold_grid[-1] for b in unique_buckets}

    for thresh in threshold_grid:
        pred_bool = probs > thresh
        for b in unique_buckets:
            cols = cols_by_bucket[b]
            p = pred_bool[:, cols]
            l = labels_by_bucket[b]
            tp = np.count_nonzero(p & l)
            fp = np.count_nonzero(p & ~l)
            fn = np.count_nonzero(~p & l)
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1[b]:
                best_f1[b] = f1
                threshold_map[b] = thresh
    return threshold_map


def apply_group_thresholds(probs, bucket_ids, threshold_map):
    """Build the final binary prediction matrix using each column's bucket threshold."""
    preds = np.zeros_like(probs, dtype=int)
    for bucket_id, thresh in threshold_map.items():
        cols = np.where(bucket_ids == bucket_id)[0]
        preds[:, cols] = (probs[:, cols] > thresh).astype(int)
    return preds


def _prf_from_counts_micro(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _prf_from_counts_macro(tp_class, fp_class, fn_class):
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp_class + fp_class > 0, tp_class / (tp_class + fp_class), 0.0)
        recall = np.where(tp_class + fn_class > 0, tp_class / (tp_class + fn_class), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision.mean(), recall.mean(), f1.mean()


def compute_prf_stats(labels_bool, preds_bool):
    """Fast replacement for sklearn's f1/precision/recall/accuracy_score - on a
    (n_samples x 22,027) matrix, sklearn's per-call cost is ~70s each (minutes per
    eval across all 6 calls); plain numpy boolean reductions do the same in ~1s."""
    tp_class = np.count_nonzero(preds_bool & labels_bool, axis=0)
    fp_class = np.count_nonzero(preds_bool & ~labels_bool, axis=0)
    fn_class = np.count_nonzero(~preds_bool & labels_bool, axis=0)

    precision_micro, recall_micro, f1_micro = _prf_from_counts_micro(
        tp_class.sum(), fp_class.sum(), fn_class.sum())
    precision_macro, recall_macro, f1_macro = _prf_from_counts_macro(tp_class, fp_class, fn_class)

    return {
        'f1_micro': f1_micro, 'f1_macro': f1_macro,
        'precision_micro': precision_micro, 'precision_macro': precision_macro,
        'recall_micro': recall_micro, 'recall_macro': recall_macro,
        'subset_accuracy': np.all(preds_bool == labels_bool, axis=1).mean(),
    }


def stratified_recall(labels_bool, preds_bool, bucket_ids, bucket_names):
    """Recall/F1 per frequency bucket - exposes the long-tail blind spot micro-F1 hides."""
    metrics = {}
    for bucket_id, name in enumerate(bucket_names):
        cols = np.where(bucket_ids == bucket_id)[0]
        if len(cols) == 0:
            continue
        l = labels_bool[:, cols]
        p = preds_bool[:, cols]
        tp = np.count_nonzero(p & l)
        fp = np.count_nonzero(p & ~l)
        fn = np.count_nonzero(~p & l)
        _, recall, f1 = _prf_from_counts_micro(tp, fp, fn)
        metrics[f'recall_freq_{name}'] = recall
        metrics[f'f1_freq_{name}'] = f1
    return metrics


def make_compute_metrics(bucket_ids, bucket_names, threshold_grid):
    """Closure so compute_metrics (fixed signature required by Trainer) can still
    access the per-class bucket assignment."""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(predictions)).numpy()

        threshold_map = find_best_threshold_per_group(probs, labels, bucket_ids, threshold_grid)
        predictions_binary = apply_group_thresholds(probs, bucket_ids, threshold_map)

        labels_bool = labels.astype(bool)
        preds_bool = predictions_binary.astype(bool)

        metrics = compute_prf_stats(labels_bool, preds_bool)
        metrics.update({
            'avg_predictions_per_sample': predictions_binary.sum(axis=1).mean(),
            'avg_true_per_sample': labels.sum(axis=1).mean(),
            'total_predictions': predictions_binary.sum(),
            'total_true': labels.sum(),
        })
        for bucket_id, thresh in threshold_map.items():
            metrics[f'threshold_freq_{bucket_names[bucket_id]}'] = thresh
        metrics.update(stratified_recall(labels_bool, preds_bool, bucket_ids, bucket_names))
        return metrics
    return compute_metrics


class WeightedBCETrainer(Trainer):
    """Trainer using a per-class pos_weight BCE loss instead of the model's
    default unweighted BCE, to counter vanishing gradients on rare tags.
    Set trainer.pos_weight after construction."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def create_sample_predictions_report(model, tokenizer, mlb, test_data, test_labels, bucket_ids, output_dir, threshold_map, num_samples=50):
    """Create a human-readable report of predictions vs actual tags.

    threshold_map must come from a fit against val labels (see main()) - fitting
    it against this function's own sample subset of test_data would score
    predictions against the same labels used to pick the decision rule."""
    model.eval()
    device = next(model.parameters()).device

    sample_indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)

    results = []
    all_predictions = []
    all_true = []

    for idx in sample_indices:
        text = test_data[idx]['text']

        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        all_predictions.append(predictions)
        all_true.append(test_labels[idx])

    all_predictions = np.array(all_predictions)
    all_true = np.array(all_true)

    predictions_binary = apply_group_thresholds(all_predictions, bucket_ids, threshold_map)
    thresholds_used = {FREQ_BIN_NAMES[b]: float(t) for b, t in threshold_map.items()}
    print(f"Per-bucket thresholds used (frozen, fit on val): {thresholds_used}")

    for i, idx in enumerate(sample_indices):
        text = test_data[idx]['text']
        true_tags = test_data[idx]['tags']
        predictions = all_predictions[i]
        predicted_labels = predictions_binary[i]

        predicted_tags = mlb.inverse_transform(predicted_labels.reshape(1, -1))[0] if predicted_labels.sum() > 0 else []

        top_indices = np.argsort(predictions)[-10:][::-1]  # Top 10 predictions
        top_scores = [(mlb.classes_[idx], float(predictions[idx])) for idx in top_indices]

        results.append({
            'id': test_data[idx].get('id'),
            'university': test_data[idx].get('university'),
            'text': text[:300] + '...' if len(text) > 300 else text,
            'true_tags': list(true_tags),
            'predicted_tags': list(predicted_tags),
            'top_predictions_with_scores': top_scores,
            'num_true': len(true_tags),
            'num_predicted': len(predicted_tags),
            'intersection': list(set(true_tags) & set(predicted_tags)),
            'precision': len(set(true_tags) & set(predicted_tags)) / len(predicted_tags) if predicted_tags else 0,
            'recall': len(set(true_tags) & set(predicted_tags)) / len(true_tags) if true_tags else 0,
        })

    # Save detailed report
    report_path = os.path.join(output_dir, 'sample_predictions_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Create summary
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    summary = {
        'total_samples': len(results),
        'thresholds_used': thresholds_used,
        'average_precision': avg_precision,
        'average_recall': avg_recall,
        'average_f1': avg_f1,
        'avg_true_tags_per_sample': np.mean([r['num_true'] for r in results]),
        'avg_predicted_tags_per_sample': np.mean([r['num_predicted'] for r in results]),
        'samples_with_predictions': sum(1 for r in results if r['num_predicted'] > 0),
        'samples_with_correct_predictions': sum(1 for r in results if len(r['intersection']) > 0)
    }
    
    summary_path = os.path.join(output_dir, 'prediction_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSample predictions report saved to: {report_path}")
    print(f"Prediction summary saved to: {summary_path}")
    print(f"Sample-level metrics - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    print(f"Samples with predictions: {summary['samples_with_predictions']}/{len(results)}")
    print(f"Samples with at least one correct prediction: {summary['samples_with_correct_predictions']}/{len(results)}")
    
    return results, summary

def main():
    # Configuration
    DATA_DIR = "boa_strangling/data/data_multiuni_2026-07-16"
    OUTPUT_DIR = f"boa_strangling/results/xlm_roberta_multilabel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    MODEL_NAME = "xlm-roberta-base"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "model"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)
    
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Initialize wandb
    wandb.init(
        project="xlm-roberta-multilabel-tagging",
        name=f"xlm_roberta_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_name": MODEL_NAME,
            "max_length": 512,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "num_epochs": 36,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "data_dir": DATA_DIR,
            "output_dir": OUTPUT_DIR
        }
    )
    
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data(DATA_DIR)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Prepare labels
    print("Preparing labels...")
    mlb, train_labels, val_labels, test_labels = prepare_labels(train_data, val_data, test_data)
    num_labels = len(mlb.classes_)
    print(f"Number of unique labels: {num_labels}")

    # Frequency-bucket assignment and loss reweighting for the long-tail label space
    bucket_ids = compute_tag_bucket_ids(train_labels)
    bucket_sizes = {FREQ_BIN_NAMES[b]: int((bucket_ids == b).sum()) for b in range(len(FREQ_BIN_NAMES))}
    print(f"Bucket sizes: {bucket_sizes}")
    pos_weight = compute_pos_weight(train_labels)
    print(f"pos_weight range: min={pos_weight.min().item():.2f}, max={pos_weight.max().item():.2f}, mean={pos_weight.mean().item():.2f}")

    # Save label encoder
    encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(mlb, f)
    print(f"Label encoder saved to: {encoder_path}")
    
    # Initialize tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    # Prepare texts
    train_texts = [item['text'] for item in train_data]
    val_texts = [item['text'] for item in val_data]
    test_texts = [item['text'] for item in test_data]
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer)
    test_dataset = MultiLabelDataset(test_texts, test_labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        num_train_epochs=36,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="wandb",
        run_name=f"xlm_roberta_multilabel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize trainer
    trainer = WeightedBCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(bucket_ids, FREQ_BIN_NAMES, THRESHOLD_GRID),
        callbacks=[]
    )
    trainer.pos_weight = pos_weight

    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save final model
    model_save_path = os.path.join(OUTPUT_DIR, "model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Evaluate on test set with thresholds frozen from validation. Fitting
    # thresholds directly on test labels (what trainer.evaluate(test_dataset)
    # did before, via compute_metrics fitting fresh on whatever split it's
    # handed) leaks test labels into the decision rule that's then scored
    # against those same labels - see THESIS_LOG.md test-threshold-leakage note.
    trainer.compute_metrics = None  # avoid the closure re-fitting thresholds inside predict()

    print("Fitting per-bucket thresholds on validation set (frozen for test evaluation)...")
    val_pred = trainer.predict(val_dataset)
    val_probs = torch.sigmoid(torch.tensor(val_pred.predictions)).numpy()
    frozen_threshold_map = find_best_threshold_per_group(val_probs, val_pred.label_ids, bucket_ids, THRESHOLD_GRID)
    frozen_thresholds = {FREQ_BIN_NAMES[b]: float(t) for b, t in frozen_threshold_map.items()}
    print(f"Frozen per-bucket thresholds (fit on val, applied to test): {frozen_thresholds}")

    print("Evaluating on test set...")
    test_pred = trainer.predict(test_dataset)
    test_probs = torch.sigmoid(torch.tensor(test_pred.predictions)).numpy()
    test_labels_arr = test_pred.label_ids
    test_preds_binary = apply_group_thresholds(test_probs, bucket_ids, frozen_threshold_map)
    test_labels_bool = test_labels_arr.astype(bool)
    test_preds_bool = test_preds_binary.astype(bool)

    test_results = compute_prf_stats(test_labels_bool, test_preds_bool)
    test_results.update({
        'avg_predictions_per_sample': test_preds_binary.sum(axis=1).mean(),
        'avg_true_per_sample': test_labels_arr.sum(axis=1).mean(),
        'total_predictions': int(test_preds_binary.sum()),
        'total_true': int(test_labels_arr.sum()),
    })
    for bucket_id, thresh in frozen_threshold_map.items():
        test_results[f'threshold_freq_{FREQ_BIN_NAMES[bucket_id]}'] = thresh
    test_results.update(stratified_recall(test_labels_bool, test_preds_bool, bucket_ids, FREQ_BIN_NAMES))
    # keep the eval_ prefix the old trainer.evaluate()-based schema used, since
    # THESIS_LOG.md/wandb logging/predict.py tooling downstream expect it
    test_results = {f'eval_{k}': v for k, v in test_results.items()}

    print("Test Results (frozen val-fit thresholds):")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save test results
    results_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    # Create sample predictions report
    print("Creating sample predictions report...")
    predictions_output_dir = os.path.join(OUTPUT_DIR, "predictions")
    sample_results, summary = create_sample_predictions_report(
        model, tokenizer, mlb, test_data, test_labels, bucket_ids, predictions_output_dir,
        frozen_threshold_map, num_samples=100
    )
    
    # Log final results to wandb
    wandb.log({
        "final_test_f1_micro": test_results.get("eval_f1_micro", 0),
        "final_test_f1_macro": test_results.get("eval_f1_macro", 0),
        "final_test_precision_micro": test_results.get("eval_precision_micro", 0),
        "final_test_recall_micro": test_results.get("eval_recall_micro", 0),
        "sample_predictions_precision": summary["average_precision"],
        "sample_predictions_recall": summary["average_recall"],
        "sample_predictions_f1": summary["average_f1"]
    })
    
    # Save configuration
    config = {
        "model_name": MODEL_NAME,
        "data_dir": DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "num_labels": num_labels,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "training_args": training_args.to_dict(),
        "test_results": test_results,
        "timestamp": datetime.now().isoformat()
    }
    
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    wandb.alert(
        title="Training completed",
        text=f"{OUTPUT_DIR}\nf1_micro={test_results.get('eval_f1_micro', 0):.4f} "
             f"f1_macro={test_results.get('eval_f1_macro', 0):.4f}",
    )
    wandb.finish()

    print(f"\nTraining completed!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Model saved in: {model_save_path}")
    print(f"Label encoder saved in: {encoder_path}")
    print(f"Sample predictions available in: {predictions_output_dir}")
    
    return OUTPUT_DIR, test_results

if __name__ == "__main__":
    try:
        output_dir, results = main()
        print(f"\nFinal test F1-micro: {results.get('eval_f1_micro', 0):.4f}")
        print(f"Final test F1-macro: {results.get('eval_f1_macro', 0):.4f}")
    except Exception as e:
        try:
            wandb.alert(title="Training crashed", text=repr(e))
            wandb.finish(exit_code=1)
        except Exception:
            pass
        raise