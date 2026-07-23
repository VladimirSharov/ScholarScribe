#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Tagging Model Training Script

This script trains a multi-label classification model for thesis tagging 
using XLM-RoBERTa and Weights & Biases for experiment tracking.
"""

import os
import logging
import pickle
import time
import json
from typing import List, Dict, Tuple

import torch
import wandb
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import precision_recall_fscore_support

# Configure logging
def setup_logging(log_dir: str = 'logs', debug: bool = False) -> logging.Logger:
    """
    Set up a comprehensive logging configuration.
    
    Args:
        log_dir (str): Directory to save log files
        debug (bool): Whether to enable debug-level logging
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate a unique log filename with timestamp
    log_filename = os.path.join(log_dir, f"thesis_tagger_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Log system and GPU information
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    
    return logger

class Config:
    """
    Centralized configuration management with enhanced flexibility.
    Allows for easy modification and logging of configuration parameters.
    """
    def __init__(self, 
                 train_path: str = "boa_strangling/data/data_split_fi_eng_min5_abs40/train.json",
                 val_path: str = "boa_strangling/data/data_split_fi_eng_min5_abs40/val.json",
                 test_path: str = "boa_strangling/data/data_split_fi_eng_min5_abs40/test.json",
                 tag_vocab_path: str = "boa_strangling/data/data_split_fi_eng_min5_abs40/tag_vocab.json",  # NEW
                 model_name: str = "xlm-roberta-base",
                 max_length: int = 512,
                 batch_size: int = 32,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 4,
                 debug: bool = False):
        self.tag_vocab_path = tag_vocab_path
        """
        Initialize configuration with flexible parameters.
        
        Args:
            debug (bool): Enable debug mode for more verbose logging
        """
        # Paths configuration
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        
        # Output configuration
        self.output_dir = f"model_embedding/model_output_mn_{time.strftime('%Y%m%d_%H%M%S')}"
        self.tokenized_data_path = "model_embedding/tokenized_datasets"
        
        # Model configuration
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Debug flag
        self.debug = debug
        
        # Setup logging
        self.logger = setup_logging(debug=debug)
        
        # Log the configuration
        self.log_configuration()
    
    def log_configuration(self):
        """Log all configuration parameters for transparency."""
        self.logger.info("Configuration Parameters:")
        for key, value in vars(self).items():
            if not key.startswith('__') and key != 'logger':
                self.logger.info(f"{key}: {value}")

class ThesisModelTrainer:
    """
    Comprehensive trainer class for thesis tagging model.
    Encapsulates data loading, preprocessing, model training, and evaluation.
    """
    def __init__(self, config: Config):
        """
        Initialize the trainer with given configuration.
        
        Args:
            config (Config): Configuration object with training parameters
        """
        self.config = config
        self.logger = config.logger

        # Ensure output directories exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        
        # Initialize wandb with dynamic project naming
        self._setup_wandb()
    
    def _setup_wandb(self):
        """
        Setup Weights & Biases logging with dynamic project and run names.
        Ensures consistency between local output and wandb tracking.
        """
        # Extract a unique identifier from the output directory
        run_name = os.path.basename(self.config.output_dir)
        
        wandb.init(
            project="thesis-tagger",  # Consistent project name
            name=run_name,  # Use the same name as local output directory
            config=vars(self.config)  # Log all configuration parameters
        )
    
    def load_datasets(self) -> DatasetDict:
        """
        Load and preprocess datasets from JSON files.
        
        Returns:
            DatasetDict: Processed train and validation datasets
        """
        self.logger.info("Loading datasets...")
        
        # Load data with error handling
        try:
            train_data = Dataset.from_pandas(pd.read_json(self.config.train_path))
            val_data = Dataset.from_pandas(pd.read_json(self.config.val_path))
        except Exception as e:
            self.logger.error(f"Error loading datasets: {e}")
            raise
        
        dataset = DatasetDict({
            'train': train_data,
            'validation': val_data,
        })
        
        
        self.logger.info(f"Loaded datasets - Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}")
        
        return dataset
    
    
    def prepare_model_and_tokenizer(self, num_labels: int):
        """
        Prepare tokenizer and model for multi-label classification.
        
        Args:
            num_labels (int): Number of unique labels
        
        Returns:
            Tuple of (tokenizer, model)
        """
        self.logger.info(f"Preparing model with {num_labels} labels")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        return tokenizer, model
    
    def train(self):
        """
        Main training method that orchestrates the entire training process.
        """
        # Load datasets
        dataset = self.load_datasets()
        
        # Load tag2id from file instead of recomputing
        with open(self.config.tag_vocab_path, "r", encoding="utf-8") as f:
            tag2id = json.load(f)
        
        # Save tag mapping
        self._save_tag_mapping(tag2id)
        
        # Prepare model and tokenizer
        tokenizer, model = self.prepare_model_and_tokenizer(len(tag2id))
        
        # Tokenize datasets
        tokenized_datasets = dataset.map(
            self._create_preprocess_function(tokenizer, tag2id), 
            batched=True
        )
        
        # Configure training arguments with more logging points
        training_args = self._create_training_arguments()
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=ThesisMetrics(tag2id).compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Finish wandb logging
        wandb.finish()
    
    def _save_tag_mapping(self, tag2id: Dict[str, int]):
        """
        Save tag to ID mapping for future reference.

        Args:
            tag2id (Dict[str, int]): Mapping of tags to their integer IDs
        """
        # Ensure the output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Construct the mapping file path
        mapping_filename = os.path.join(
            self.config.output_dir,
            f"tag_mapping_{time.strftime('%Y%m%d_%H%M%S')}.pickle"
        )

        # Save the tag-to-ID mapping as a pickle file
        with open(mapping_filename, 'wb') as f:
            pickle.dump(tag2id, f, pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Tag mapping saved to {mapping_filename}")
    
    def _create_preprocess_function(self, tokenizer, tag2id):
        """
        Create a preprocessing function for tokenization.
        
        Args:
            tokenizer: Tokenizer to use for preprocessing
            tag2id (Dict): Mapping of tags to their integer IDs
        
        Returns:
            Callable: Preprocessing function
        """
        def preprocess_function(examples):
            # Tokenize
            tokenized_inputs = tokenizer(
                examples["clean_full_text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )

            # Labels: safe lookup, skip missing tags, fill with dummy if needed
            labels = []
            for tags in examples["combined_tags"]:
                valid_tags = [tag2id[tag] for tag in tags if tag in tag2id]
                # Avoid empty lists (replace with dummy label, e.g., 0)
                if not valid_tags:
                    valid_tags = [0]
                labels.append(valid_tags)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        return preprocess_function
    
    def _create_training_arguments(self):
        """
        Create training arguments with enhanced logging and evaluation.
        
        Returns:
            TrainingArguments: Configured training arguments
        """
        return TrainingArguments(
            output_dir=self.config.output_dir,
            lr_scheduler_type="constant",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=0.01,
            # load_best_model_at_end=True,
            # metric_for_best_model="loss",
            report_to="wandb",
            logging_strategy="steps",
            logging_steps=10,  # More frequent logging (default was "epoch")
            logging_dir="./logs",
            eval_steps=50,     # Evaluate more frequently
            save_total_limit=3,  # Keep only the last 3 model checkpoints
            fp16=True,
        )
    
    @staticmethod
    def _get_first_abstract(abstracts: List[str]) -> str:
        """
        Safely extract the first abstract.
        
        Args:
            abstracts (List[str]): List of abstracts
        
        Returns:
            str: First abstract or empty string
        """
        return abstracts[0] if abstracts else ""

class ThesisMetrics:
    """
    Metrics computation for multi-label classification.
    Provides comprehensive performance evaluation.
    """
    def __init__(self, tag2id: Dict[str, int], threshold: float = 0.5):
        """
        Initialize metrics computation.
        
        Args:
            tag2id (Dict[str, int]): Mapping of tags to their integer IDs
            threshold (float): Threshold for binary classification
        """
        self.tag2id = tag2id
        self.id2tag = {v: k for k, v in tag2id.items()}
        self.threshold = threshold
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """
        Compute comprehensive metrics for multi-label classification.
        
        Args:
            eval_pred (Tuple): Predictions and true labels
        
        Returns:
            Dict: Computed metrics
        """
        predictions, labels = eval_pred
        pred_labels = (predictions > self.threshold).astype(int)
        
        # Precision, Recall, F1 (macro-averaged)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='macro', zero_division=0
        )
        
        confusion_metrics = self._compute_confusion_matrix(pred_labels, labels)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            **confusion_metrics
        }
        
        return metrics
    
    def _compute_confusion_matrix(self, pred_labels: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        Compute normalized confusion matrix metrics.
        
        Args:
            pred_labels (np.ndarray): Predicted labels
            true_labels (np.ndarray): True labels
        
        Returns:
            Dict: Normalized confusion matrix metrics
        """
        metric = {
            'true_negative': 0, 
            'false_positive': 0, 
            'false_negative': 0, 
            'true_positive': 0
        }
        total_elements = 0
        
        # Calculate confusion matrix
        for pred_row, true_row in zip(pred_labels, true_labels):
            for pred, true in zip(pred_row, true_row):
                if not pred and not true:
                    metric['true_negative'] += 1
                elif pred and not true:
                    metric['false_positive'] += 1
                elif not pred and true:
                    metric['false_negative'] += 1
                elif pred and true:
                    metric['true_positive'] += 1
                total_elements += 1
        
        # Normalize confusion matrix metrics by total elements
        normalized_metrics = {
            key: value / total_elements 
            for key, value in metric.items()
        }
        
        return normalized_metrics

def main():
    """
    Main entry point for the script.
    Allows for easy configuration and debugging.
    """
    # Parse command-line arguments or use default configuration
    config = Config(debug=True)  # Set debug=True for verbose logging

    # Initialize trainer with the configuration
    trainer = ThesisModelTrainer(config)

    # Train the model
    try:
        trainer.train()
    except Exception as e:
        trainer.logger.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()
