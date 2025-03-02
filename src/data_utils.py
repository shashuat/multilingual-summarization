#!/usr/bin/env python3
"""
data_utils.py - Utilities for data processing and preprocessing

This module provides functions for loading, preprocessing, and preparing
datasets for model training and evaluation.
"""

import os
from typing import Dict, Any, Callable, Optional, Tuple
from datasets import load_from_disk, Dataset, DatasetDict


def load_dataset_split(
    dataset_path: str,
    language: str
) -> DatasetDict:
    """
    Load a dataset for a specific language from disk
    
    Args:
        dataset_path: Base path to the dataset
        language: Language to load
    
    Returns:
        dataset: DatasetDict with train/validation/test splits
    """
    dataset_dir = os.path.join(dataset_path, language)
    print(f"Loading dataset from {dataset_dir}...")
    
    try:
        dataset = load_from_disk(dataset_dir)
        print(f"Dataset splits: {dataset.keys()}")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['validation'])}")
        print(f"Test examples: {len(dataset['test'])}")
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")


def filter_short_samples(
    dataset: Dataset, 
    min_text_length: int = 100, 
    min_summary_length: int = 10
) -> Dataset:
    """
    Filter out samples with text or summary that are too short
    
    Args:
        dataset: Dataset to filter
        min_text_length: Minimum text length to keep
        min_summary_length: Minimum summary length to keep
    
    Returns:
        filtered_dataset: Filtered dataset
    """
    def is_valid(example):
        return (len(example["text"]) >= min_text_length and 
                len(example["summary"]) >= min_summary_length)
    
    return dataset.filter(is_valid)


def trim_long_summaries(
    dataset: Dataset, 
    max_summary_length: int = 2000
) -> Dataset:
    """
    Trim summaries that are too long
    
    Args:
        dataset: Dataset with summaries to trim
        max_summary_length: Maximum summary length
    
    Returns:
        trimmed_dataset: Dataset with trimmed summaries
    """
    def trim_summary(example):
        example["summary"] = example["summary"][:max_summary_length]
        return example
    
    return dataset.map(trim_summary)


def prepare_dataset_for_training(
    dataset: DatasetDict,
    preprocess_function: Callable,
    column_names: Optional[list] = None
) -> DatasetDict:
    """
    Prepare dataset for training by applying preprocessing and tokenization
    
    Args:
        dataset: DatasetDict with splits
        preprocess_function: Function to apply to examples
        column_names: Columns to remove after preprocessing (None for all)
    
    Returns:
        processed_dataset: Dictionary of processed datasets by split
    """
    print("Preprocessing dataset...")
    processed_dataset = {}
    
    for split, ds in dataset.items():
        ds = filter_short_samples(trim_long_summaries(ds))
        
        # Apply preprocessing function
        remove_cols = column_names if column_names else ds.column_names
        processed_dataset[split] = ds.map(
            preprocess_function,
            batched=True,
            remove_columns=remove_cols
        )
    
    # Print stats about processed dataset
    print("Processed dataset statistics:")
    for split, ds in processed_dataset.items():
        print(f"  {split}: {len(ds)} examples")
    
    return processed_dataset


def create_preprocess_function(
    tokenizer: Any,
    max_input_length: int,
    max_target_length: int,
    prefix: str = ""
) -> Callable:
    """
    Create a preprocessing function for dataset tokenization
    
    Args:
        tokenizer: Tokenizer to use
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        prefix: Optional prefix to add to input texts
    
    Returns:
        preprocess_fn: Function to preprocess examples
    """
    def preprocess_function(examples):
        # Prepare inputs
        inputs = [prefix + text for text in examples["text"]]
        targets = examples["summary"]
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            padding="max_length",
            truncation=True
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                padding="max_length",
                truncation=True
            )
        
        # Replace padding token id with -100 so it's ignored in the loss
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        
        return model_inputs
    
    return preprocess_function