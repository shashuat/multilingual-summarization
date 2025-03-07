#!/usr/bin/env python3
"""
finetune_llama.py - Fine-tune Llama-3.2-1B-Instruct for summarization using LoRA
"""

import os
import argparse
import torch
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from tqdm import tqdm

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prompt(text, summary=""):
    """Create a prompt for summarization task"""
    if summary:
        # For training examples with labels
        return f"""### Instruction:
Summarize the following text in a concise manner.

### Input:
{text}

### Response:
{summary}"""
    else:
        # For inference
        return f"""### Instruction:
Summarize the following text in a concise manner.

### Input:
{text}

### Response:
"""


def preprocess_dataset(dataset, tokenizer, max_length=1024, max_target_length=256):
    """Preprocess dataset for training a causal language model"""
    
    def process_examples(examples):
        inputs = []
        for text, summary in zip(examples["text"], examples["summary"]):
            inputs.append(create_prompt(text, summary))
        
        # Tokenize inputs
        tokenized = tokenizer(
            inputs,
            max_length=max_length + max_target_length,  # Allow space for the summary
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        return tokenized
    
    # Process the dataset
    processed_dataset = dataset.map(
        process_examples,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset"
    )
    
    return processed_dataset


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2-1B-Instruct for summarization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name or path")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the HuggingFace dataset directory")
    parser.add_argument("--language", type=str, required=True,
                        help="Language code (e.g., 'en', 'fr')")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for saving model")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum input sequence length")
    parser.add_argument("--max_target_length", type=int, default=256,
                        help="Maximum target sequence length")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA attention dropout")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load model and tokenizer
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    
    # Use 4-bit quantization for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Step 2: Set up LoRA
    logger.info(f"Setting up LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Step 3: Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}/{args.language}")
    dataset = load_from_disk(os.path.join(args.dataset_path, args.language))
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    # Step 4: Preprocess dataset
    logger.info("Preprocessing datasets")
    train_dataset = preprocess_dataset(
        train_dataset, 
        tokenizer, 
        max_length=args.max_length, 
        max_target_length=args.max_target_length
    )
    
    val_dataset = preprocess_dataset(
        val_dataset, 
        tokenizer, 
        max_length=args.max_length, 
        max_target_length=args.max_target_length
    )
    
    # Step 5: Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
    )
    
    # Step 6: Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb reporting
        remove_unused_columns=False,  # Required for LLM fine-tuning
        ddp_find_unused_parameters=False,
    )
    
    # Step 7: Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Step 8: Train and save model
    logger.info("Starting training")
    trainer.train()
    
    # Step 9: Save the model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()