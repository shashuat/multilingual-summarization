#!/usr/bin/env python3
"""
model_utils.py - Utilities for loading, saving, and configuring models

This module provides functions for setting up models and tokenizers for both
fine-tuning and evaluation processes.
"""

import os
import torch
from typing import Tuple, Optional, Dict, Any

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig


def load_base_model(
    model_name: str,
    use_fp16: bool = True
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load a base model and tokenizer from HuggingFace
    
    Args:
        model_name: Name or path of the model to load
        use_fp16: Whether to use FP16 for model weights
    
    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    print(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer


def apply_lora(
    model: AutoModelForSeq2SeqLM,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> AutoModelForSeq2SeqLM:
    """
    Apply LoRA configuration to a model for fine-tuning
    
    Args:
        model: Base model to apply LoRA to
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: List of module names to apply LoRA to (None for default)
    
    Returns:
        model: Model with LoRA configuration applied
    """
    print(f"Setting up LoRA with r={lora_r}, alpha={lora_alpha}")
    
    # Use default target modules if none provided
    if target_modules is None:
        target_modules = ["q", "v", "k", "o", "wi", "wo"]
    
    # Create LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


def load_finetuned_model(
    model_path: str,
    base_model: Optional[str] = None,
    is_lora: bool = False
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load a fine-tuned model, handling LoRA adapters if needed
    
    Args:
        model_path: Path to the fine-tuned model or adapter
        base_model: Path to base model (required for LoRA)
        is_lora: Whether this is a LoRA adapter
    
    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    if is_lora:
        if not base_model:
            # Try to get base model from config
            config_path = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        import json
                        config = json.load(f)
                    base_model = config.get("base_model_name_or_path")
                except Exception as e:
                    print(f"Error reading adapter config: {e}")
            
            if not base_model:
                raise ValueError("For LoRA models, base_model must be provided or found in adapter_config.json")
        
        # Load base model and apply LoRA adapter
        model, tokenizer = load_base_model(base_model)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Load standard fine-tuned model
        model, tokenizer = load_base_model(model_path)
    
    return model, tokenizer


def save_model_and_config(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    training_config: Dict[str, Any]
) -> None:
    """
    Save model, tokenizer, and training configuration
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
        training_config: Training configuration to save
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training configuration
    import json
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)