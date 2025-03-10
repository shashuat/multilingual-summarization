#!/usr/bin/env python3
"""
finetune_phi4.py - Fine-tune Microsoft Phi-4-mini-instruct for multilingual summarization using LoRA
with proper Phi-4 chat format, optimized parameters, and wandb logging
"""

import os
import argparse
import torch
import wandb
import json
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from tqdm import tqdm

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_system_message(language="en"):
    """Get the appropriate system message based on language"""
    if language == "en":
        return """You are an expert summarizer. Your task is to create concise, comprehensive summaries 
that capture all key points and main arguments while avoiding unnecessary details. 
Do not simply copy the first few sentences of the article. Create a summary that stands 
on its own and covers the entire article's important content."""
    elif language == "fr":
        return """Vous êtes un expert en résumé. Votre tâche est de créer des résumés concis et complets 
qui capturent tous les points clés et les arguments principaux tout en évitant les détails inutiles. 
Ne copiez pas simplement les premières phrases de l'article. Créez un résumé qui se tient 
par lui-même et couvre tout le contenu important de l'article."""
    elif language == "de":
        return """Sie sind ein Experte für Zusammenfassungen. Ihre Aufgabe ist es, prägnante, umfassende Zusammenfassungen 
zu erstellen, die alle Schlüsselpunkte und Hauptargumente erfassen und dabei unnötige Details vermeiden. 
Kopieren Sie nicht einfach die ersten Sätze des Artikels. Erstellen Sie eine Zusammenfassung, die 
eigenständig ist und den gesamten wichtigen Inhalt des Artikels abdeckt."""
    elif language == "ja":
        return """あなたは要約の専門家です。あなたの仕事は、不必要な詳細を避けながら、すべての重要なポイントと主な議論を
捉えた簡潔で包括的な要約を作成することです。記事の最初の文をそのままコピーしないでください。
それ自体で成り立ち、記事の重要な内容全体をカバーする要約を作成してください。"""
    else:
        # Default for other languages
        return f"""You are an expert summarizer. Your task is to create concise, comprehensive summaries 
in {language} that capture all key points and main arguments while avoiding unnecessary details. 
Do not simply copy the first few sentences of the article. Create a summary that stands 
on its own and covers the entire article's important content."""


def create_prompt(text, language="en", summary=""):
    """Create a language-specific prompt for summarization task using Phi-4 chat format"""
    
    # Get appropriate system message
    system_message = get_system_message(language)
    
    # Create user message (the article to summarize)
    user_message = f"Please summarize the following article: {text}"
    
    # Format using Phi-4's chat format
    if summary:
        # For training examples with labels
        prompt = f"<|system|>{system_message}<|end|><|user|>{user_message}<|end|><|assistant|>{summary}"
    else:
        # For inference
        prompt = f"<|system|>{system_message}<|end|><|user|>{user_message}<|end|><|assistant|>"
    
    return prompt


def preprocess_dataset(dataset, tokenizer, language="en", max_length=3072, max_target_length=512):
    """Preprocess dataset for training a causal language model with proper labels"""
    
    def process_examples(examples):
        inputs = []
        labels = []
        
        for text, summary in zip(examples["text"], examples["summary"]):
            # Get system message for this language
            system_message = get_system_message(language)
            
            # Create user message
            user_message = f"Please summarize the following article: {text}"
            
            # Format the full prompt with summary (for input_ids)
            full_prompt = f"<|system|>{system_message}<|end|><|user|>{user_message}<|end|><|assistant|>{summary}"
            
            # Format just the prefix (for creating labels with -100s)
            prefix_prompt = f"<|system|>{system_message}<|end|><|user|>{user_message}<|end|><|assistant|>"
            
            # Tokenize the full prompt
            tokenized_full = tokenizer(
                full_prompt,
                max_length=max_length + max_target_length,
                padding=False,
                truncation=True,
                return_tensors=None
            )
            
            # Tokenize just the prefix part (without the summary)
            tokenized_prefix = tokenizer(
                prefix_prompt,
                max_length=max_length,
                padding=False,
                truncation=True,
                return_tensors=None
            )
            
            # Create label tokens: -100 for prefix (ignored in loss) and actual tokens for summary
            label_tokens = [-100] * len(tokenized_prefix["input_ids"]) + tokenized_full["input_ids"][len(tokenized_prefix["input_ids"]):]
            
            # Ensure both have the same length (pad with -100 if needed)
            if len(label_tokens) < len(tokenized_full["input_ids"]):
                label_tokens += [-100] * (len(tokenized_full["input_ids"]) - len(label_tokens))
            else:
                label_tokens = label_tokens[:len(tokenized_full["input_ids"])]
            
            inputs.append(tokenized_full["input_ids"])
            labels.append(label_tokens)
        
        return {
            "input_ids": inputs,
            "attention_mask": [
                [1] * len(input_ids) for input_ids in inputs
            ],
            "labels": labels
        }
    
    # Process the dataset
    processed_dataset = dataset.map(
        process_examples,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset"
    )
    
    return processed_dataset


# Custom callback to log more data to W&B
class WandbMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and wandb.run is not None:
            # Log detailed metrics
            wandb.log(metrics, step=state.global_step)
            
            # Log example count
            wandb.log({"train/example_count": state.global_step * args.train_batch_size * args.gradient_accumulation_steps}, 
                      step=state.global_step)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune Phi-4-mini-instruct for multilingual summarization")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-4-mini-instruct",
                        help="Model name or path (default: microsoft/Phi-4-mini-instruct)")
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
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--max_length", type=int, default=3072,
                        help="Maximum input sequence length (default: 3072)")
    parser.add_argument("--max_target_length", type=int, default=512,
                        help="Maximum target sequence length (default: 512)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA attention dropout (default: 0.05)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer (default: 0.01)")
    parser.add_argument("--wandb_project", type=str, default="mulsum-phi",
                        help="Weights & Biases project name (default: mulsum-phi)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (default: None)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (default: None)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb for experiment tracking
    run_name = args.wandb_run_name or f"phi4-{args.language}-r{args.lora_r}-lr{args.learning_rate}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        entity=args.wandb_entity,
        config={
            "model_name": args.model_name,
            "language": args.language,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps, 
            "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_length": args.max_length,
            "max_target_length": args.max_target_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
        }
    )
    
    # Step 1: Load model and tokenizer
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    
    # Use 4-bit quantization for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Log BnB config to wandb
    wandb.config.update({"quantization": "4-bit", "bnb_compute_dtype": "bfloat16"})
    
    # Load tokenizer with special tokens for Phi-4
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure the special tokens are properly set for Phi-4
    special_tokens = {
        "pad_token": tokenizer.eos_token,
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model with quantization and set use_cache=False for gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False  # Explicitly set use_cache=False
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Step 2: Set up LoRA
    logger.info(f"Setting up LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Target modules for Phi-4 architecture
    target_modules = [
        "Wqkv", "out_proj",  # Phi-4 attention modules
        "up_proj", "down_proj"  # Phi-4 MLP modules
    ]
    
    # Log target modules to wandb
    wandb.config.update({"target_modules": target_modules})
    
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
    trainable_params, all_params = model.get_nb_trainable_parameters()
    
    # Log trainable parameters info
    logger.info(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
    wandb.config.update({
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": 100 * trainable_params / all_params
    })
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # Step 3: Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}/{args.language}")
    dataset = load_from_disk(os.path.join(args.dataset_path, args.language))
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    wandb.config.update({
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset)
    })
    
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
    
    # Step 5: Create data collator that handles padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
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
        logging_steps=50,  # Log more frequently for better wandb plots
        eval_strategy="epoch",  
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        load_best_model_at_end=True,
        report_to="wandb",  # Enable wandb reporting
        remove_unused_columns=False,  # Required for custom datasets
        ddp_find_unused_parameters=False,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        group_by_length=True,  # Group sequences of similar length to reduce padding
        fp16=False,  # Use bf16 instead of fp16 for better numerical stability
        gradient_checkpointing=True,  # Enable gradient checkpointing
        # Add label names (for internal handling)
        label_names=["labels"],
        # Fix deepspeed compatibility
        deepspeed=None,
        # Optimize for training speed
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Add a run name for wandb
        run_name=run_name,
    )
    
    # Step 7: Create trainer with WandB callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[WandbMetricsCallback()],
    )
    
    # Log a sample prompt for reference
    sample_text = train_dataset[0]["input_ids"]
    sample_text_decoded = tokenizer.decode(sample_text)
    wandb.config.update({"sample_prompt": sample_text_decoded[:500] + "..."})  # First 500 chars
    
    # Step 8: Train and save model
    logger.info("Starting training")
    trainer.train()
    
    # Step 9: Save the model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Log model size after training
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    wandb.log({"model_size_mb": model_size_mb})
    
    # Save example predictions for the validation set
    logger.info("Generating example predictions for logging")
    try:
        example_predictions = []
        # Generate predictions for a few examples
        for i in range(min(5, len(val_dataset))):
            # Get example input
            example_input = val_dataset[i]["input_ids"]
            # Find the assistant token position
            assistant_start = None
            for j, token_id in enumerate(example_input):
                if tokenizer.decode([token_id]) == "<|assistant|>":
                    assistant_start = j
                    break
            
            if assistant_start:
                # Create input for generation (just up to the assistant token)
                gen_input = example_input[:assistant_start+1]
                # Generate continuation
                with torch.no_grad():
                    # Move input to model device
                    input_tensor = torch.tensor([gen_input]).to(model.device)
                    outputs = model.generate(
                        input_tensor,
                        max_new_tokens=args.max_target_length,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                    )
                    
                # Decode the output
                generated_text = tokenizer.decode(outputs[0][assistant_start+1:], skip_special_tokens=True)
                
                # Get the expected text (ground truth)
                expected_text = ""
                for j, token_id in enumerate(example_input[assistant_start+1:]):
                    if token_id != -100:  # Skip masked tokens
                        expected_text += tokenizer.decode([token_id])
                
                # Add to examples
                example_predictions.append({
                    "input": tokenizer.decode(example_input[:assistant_start]),
                    "generated": generated_text,
                    "expected": expected_text
                })
        
        # Log examples as a table in wandb
        if example_predictions:
            prediction_table = wandb.Table(columns=["input", "generated", "expected"])
            for ex in example_predictions:
                prediction_table.add_data(ex["input"], ex["generated"], ex["expected"])
            wandb.log({"prediction_examples": prediction_table})
            
    except Exception as e:
        logger.warning(f"Error generating example predictions: {e}")
    
    # Finish wandb run
    wandb.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()