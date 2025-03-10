#!/usr/bin/env python3
"""
finetune_llama.py - Fine-tune Llama models for multilingual summarization using LoRA
with improved instruction prompts and optimized parameters
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


def create_prompt(text, language="en", summary=""):
    """Create a language-specific prompt for summarization task"""
    if summary:
        # For training examples with labels
        if language == "en":
            prompt = f"""### Instruction:
            Please provide a concise summary of the following article in English. 
            The summary should be comprehensive, capturing all key points and main arguments, 
            but avoid unnecessary details. Do not simply copy the first few sentences of the article.

            ### Input:
            {text}

            ### Response:
            {summary}"""
        elif language == "fr":
            prompt = f"""### Instruction:
            Veuillez fournir un résumé concis de l'article suivant en français. 
            Le résumé doit être complet, capturant tous les points clés et les arguments principaux, 
            mais évitant les détails inutiles. Ne copiez pas simplement les premières phrases de l'article.

            ### Input:
            {text}

            ### Response:
            {summary}"""
        elif language == "de":
            prompt = f"""### Instruction:
            Bitte erstellen Sie eine prägnante Zusammenfassung des folgenden Artikels auf Deutsch. 
            Die Zusammenfassung sollte umfassend sein, alle Hauptpunkte und Hauptargumente erfassen, 
            aber unnötige Details vermeiden. Kopieren Sie nicht einfach die ersten Sätze des Artikels.

            ### Input:
            {text}

            ### Response:
            {summary}"""
        elif language == "ja":
            prompt = f"""### Instruction:
            以下の記事を日本語で簡潔に要約してください。
            要約は包括的であり、すべての重要なポイントと主な議論を捉える必要がありますが、
            不必要な詳細は避けてください。記事の最初の文をそのままコピーしないでください。

            ### Input:
            {text}

            ### Response:
            {summary}"""
        else:
            # Default for other languages
            prompt = f"""### Instruction:
            Please provide a concise summary of the following article in {language}. 
            The summary should be comprehensive, capturing all key points and main arguments, 
            but avoid unnecessary details. Do not simply copy the first few sentences of the article.

            ### Input:
            {text}

            ### Response:
            {summary}"""
    else:
        # For inference
        if language == "en":
            prompt = f"""### Instruction:
            Please provide a concise summary of the following article in English. 
            The summary should be comprehensive, capturing all key points and main arguments, 
            but avoid unnecessary details. Do not simply copy the first few sentences of the article.

            ### Input:
            {text}

            ### Response:
            """
        elif language == "fr":
            prompt = f"""### Instruction:
            Veuillez fournir un résumé concis de l'article suivant en français. 
            Le résumé doit être complet, capturant tous les points clés et les arguments principaux, 
            mais évitant les détails inutiles. Ne copiez pas simplement les premières phrases de l'article.

            ### Input:
            {text}

            ### Response:
            """
        elif language == "de":
            prompt = f"""### Instruction:
            Bitte erstellen Sie eine prägnante Zusammenfassung des folgenden Artikels auf Deutsch. 
            Die Zusammenfassung sollte umfassend sein, alle Hauptpunkte und Hauptargumente erfassen, 
            aber unnötige Details vermeiden. Kopieren Sie nicht einfach die ersten Sätze des Artikels.

            ### Input:
            {text}

            ### Response:
            """
        elif language == "ja":
            prompt = f"""### Instruction:
            以下の記事を日本語で簡潔に要約してください。
            要約は包括的であり、すべての重要なポイントと主な議論を捉える必要がありますが、
            不必要な詳細は避けてください。記事の最初の文をそのままコピーしないでください。

            ### Input:
            {text}

            ### Response:
            """
        else:
            # Default for other languages
            prompt = f"""### Instruction:
            Please provide a concise summary of the following article in {language}. 
            The summary should be comprehensive, capturing all key points and main arguments, 
            but avoid unnecessary details. Do not simply copy the first few sentences of the article.

            ### Input:
            {text}

            ### Response:
            """
    
    return prompt


def preprocess_dataset(dataset, tokenizer, language="en", max_length=4096, max_target_length=512):
    """Preprocess dataset for training a causal language model"""
    
    def process_examples(examples):
        inputs = []
        for text, summary in zip(examples["text"], examples["summary"]):
            inputs.append(create_prompt(text, language, summary))
        
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
    parser = argparse.ArgumentParser(description="Fine-tune Llama models for multilingual summarization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-8B-Instruct",
                        help="Model name or path (default: meta-llama/Llama-3.2-8B-Instruct)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the HuggingFace dataset directory")
    parser.add_argument("--language", type=str, required=True,
                        help="Language code (e.g., 'en', 'fr', 'de', 'ja')")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for saving model")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Training batch size (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum input sequence length (default: 4096)")
    parser.add_argument("--max_target_length", type=int, default=512,
                        help="Maximum target sequence length (default: 512)")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA attention dimension (default: 32)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha parameter (default: 64)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA attention dropout (default: 0.05)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer (default: 0.01)")
    
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
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Step 2: Set up LoRA
    logger.info(f"Setting up LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Target modules depend on the model architecture
    # For Llama models, target the attention modules and MLP
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
        "gate_proj", "up_proj", "down_proj"       # MLP modules
    ]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
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
        language=args.language,
        max_length=args.max_length, 
        max_target_length=args.max_target_length
    )
    
    val_dataset = preprocess_dataset(
        val_dataset, 
        tokenizer, 
        language=args.language,
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
        logging_steps=250,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb reporting
        remove_unused_columns=False,  # Required for LLM fine-tuning
        ddp_find_unused_parameters=False,
        lr_scheduler_type="cosine",  # Use cosine scheduler
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        group_by_length=True,  # Group sequences of similar length to reduce padding
        fp16=False,  # Use bf16 instead of fp16 for better numerical stability
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