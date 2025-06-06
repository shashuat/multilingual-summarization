#!/usr/bin/env python3
"""
finetune_llama32.py - Fine-tune Meta Llama-3.2-1B-Instruct for multilingual summarization using QLoRA
with simple prompt format matching the generate_summaries.py approach, optimized parameters, and wandb logging
Includes improved techniques: dynamic LoRA targeting, Flash Attention 2, SFT procedures, and NEFTune
"""

import os
import argparse
import torch
import wandb
import json
import numpy as np
import random
import math
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback
)
from tqdm import tqdm

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import rouge for metrics calculation
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    logger.warning("Rouge package not found. Install with 'pip install rouge' for ROUGE metrics.")
    ROUGE_AVAILABLE = False


def create_summary_prompt(text: str, language: str, summary: str = "") -> str:
    """
    Create a language-specific prompt for summarization matching generate_summaries.py approach
    If summary is provided, it's appended for training examples
    """
    # Truncate input if needed (as in generate_summaries.py)
    max_input_length = 4096
    if len(text) > max_input_length:
        text = text[:max_input_length]
    
    # Create appropriate prompt based on language (from generate_summaries.py)
    if language == "en":
        prompt = f"""Please provide a concise summary of the following article in English. 
        The summary should be comprehensive, capturing all key points and main arguments, 
        but avoid unnecessary details. Output only the summary.

        Article:
        {text}

        Summary:"""

    elif language == "fr":
        prompt = f"""Veuillez fournir un résumé concis de l'article suivant en français. 
        Le résumé doit être complet, capturant tous les points clés et les arguments principaux, 
        mais évitant les détails inutiles. Ne produisez que le résumé.

        Article:
        {text}

        Résumé:"""

    elif language == "de":
        prompt = f"""Bitte erstellen Sie eine prägnante Zusammenfassung des folgenden Artikels auf Deutsch. 
        Die Zusammenfassung sollte umfassend sein, alle Hauptpunkte und Hauptargumente erfassen, 
        aber unnötige Details vermeiden. Geben Sie nur die Zusammenfassung aus.

        Artikel:
        {text}

        Zusammenfassung:"""
        
    elif language == "ja":
        prompt = f"""以下の記事を日本語で簡潔に要約してください。
        要約は包括的であり、すべての重要なポイントと主な議論を捉える必要がありますが、
        不必要な詳細は避けてください。要約のみを出力してください。

        記事:
        {text}

        要約:"""

    elif language == "ru":
        prompt = f"""Пожалуйста, предоставьте краткое изложение следующей статьи на русском языке. 
        Резюме должно быть всеобъемлющим, охватывающим все ключевые моменты и основные аргументы, 
        но избегающим ненужных деталей. Выводите только резюме.

        Статья:
        {text}

        Резюме:"""

    else:
        # Default for other languages
        prompt = f"""Please provide a concise summary of the following article in {language}. 
        The summary should be comprehensive, capturing all key points and main arguments, 
        but avoid unnecessary details. Output only the summary.

        Article:
        {text}

        Summary:"""
    
    # For training examples, add the summary after the prompt
    if summary:
        prompt = prompt + summary
    
    return prompt


def preprocess_dataset(dataset, tokenizer, language="en", max_length=4096, max_target_length=256):
    """Preprocess dataset for training a causal language model with proper labels"""
    
    def process_examples(examples):
        inputs = []
        labels = []
        
        for text, summary in zip(examples["text"], examples["summary"]):
            # Create prompt without summary to determine where labels start
            prefix_prompt = create_summary_prompt(text, language)
            
            # Create full prompt with summary (for input_ids)
            full_prompt = create_summary_prompt(text, language, summary)
            
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


def calculate_rouge_scores(reference, generated):
    """Calculate ROUGE scores between reference and generated text"""
    if not ROUGE_AVAILABLE:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(generated, reference)[0]
        
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
    except Exception as e:
        logger.warning(f"Error calculating ROUGE scores: {e}")
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        }


def generate_model_summary(model, tokenizer, text, language, max_length=4096, max_new_tokens=256):
    """Generate a summary using the model"""
    try:
        # Create prompt
        prompt = create_summary_prompt(text, language)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                attention_mask=inputs.attention_mask
            )
            
        # Decode the output - extract just the generated part
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_output[len(prompt):].strip()
        
        return generated_text
    except Exception as e:
        logger.error(f"Error in summary generation: {e}")
        return ""


def find_linear_layers(model):
    """Find all linear layer names in the model."""
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            linear_layers.append(names[0] if len(names) == 1 else names[-1])
    
    # Remove lm_head from LoRA targeting
    if 'lm_head' in linear_layers:
        linear_layers.remove('lm_head')
    return list(set(linear_layers))


# Custom callback to calculate and log ROUGE metrics
class RougeEvaluationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset, raw_dataset, language, max_samples=5):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.raw_dataset = raw_dataset
        self.language = language
        self.max_samples = min(max_samples, len(raw_dataset))
        self.best_rouge = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Calculate ROUGE scores on validation set after each evaluation"""
        if not ROUGE_AVAILABLE:
            logger.warning("Rouge package not available. Skipping ROUGE evaluation.")
            return
        
        logger.info("Calculating ROUGE metrics on validation set...")
        
        # Take a sample of validation examples - Convert to Python int
        indices = list(range(len(self.raw_dataset)))
        sample_indices = np.random.choice(
            indices, 
            min(self.max_samples, len(self.raw_dataset)), 
            replace=False
        )
        
        rouge_scores = []
        examples = []
        
        # Temporarily set model to eval mode
        training = self.model.training
        self.model.eval()
        
        for idx in sample_indices:
            # Convert numpy.int64 to Python int
            example = self.raw_dataset[int(idx)]
            reference_summary = example["summary"]
            
            # Generate summary
            generated_summary = generate_model_summary(
                self.model,
                self.tokenizer,
                example["text"],
                self.language
            )
            
            if not generated_summary:
                continue
                
            # Calculate ROUGE scores
            scores = calculate_rouge_scores(reference_summary, generated_summary)
            rouge_scores.append(scores)
            
            # Save a few examples for display
            if len(examples) < 5:
                examples.append({
                    "text": example["text"][:300] + "...",  # Truncate for display
                    "reference": reference_summary,
                    "generated": generated_summary,
                    "rouge-1": scores["rouge-1"],
                    "rouge-2": scores["rouge-2"],
                    "rouge-l": scores["rouge-l"]
                })
        
        # Restore model's training mode
        self.model.train(training)
        
        if not rouge_scores:
            logger.warning("No valid ROUGE scores calculated.")
            return
            
        # Calculate average scores
        avg_scores = {
            metric: np.mean([score[metric] for score in rouge_scores])
            for metric in ["rouge-1", "rouge-2", "rouge-l"]
        }
        
        # Log to wandb if available
        if wandb.run is not None:
            # Log average scores
            wandb.log({
                f"rouge/val_{metric}": score
                for metric, score in avg_scores.items()
            }, step=state.global_step)
            
            # Create and log a table of examples
            if examples:
                example_table = wandb.Table(
                    columns=["text", "reference", "generated", "rouge-1", "rouge-2", "rouge-l"]
                )
                
                for ex in examples:
                    example_table.add_data(
                        ex["text"], 
                        ex["reference"], 
                        ex["generated"],
                        f"{ex['rouge-1']:.4f}",
                        f"{ex['rouge-2']:.4f}",
                        f"{ex['rouge-l']:.4f}"
                    )
                
                wandb.log({"rouge/examples": example_table}, step=state.global_step)
        
        # Print summary of ROUGE scores
        logger.info(f"ROUGE-1: {avg_scores['rouge-1']:.4f}")
        logger.info(f"ROUGE-2: {avg_scores['rouge-2']:.4f}")
        logger.info(f"ROUGE-L: {avg_scores['rouge-l']:.4f}")
        
        # Track best scores
        improved = False
        for metric in ["rouge-1", "rouge-2", "rouge-l"]:
            if avg_scores[metric] > self.best_rouge[metric]:
                self.best_rouge[metric] = avg_scores[metric]
                improved = True
        
        if improved and wandb.run is not None:
            wandb.log({
                f"rouge/best_{metric}": score
                for metric, score in self.best_rouge.items()
            }, step=state.global_step)
            
            logger.info("New best ROUGE scores!")


# Custom callback to log more data to W&B
class WandbMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and wandb.run is not None:
            # Log detailed metrics
            wandb.log(metrics, step=state.global_step)
            
            # Log example count
            wandb.log({
                "train/example_count": state.global_step * args.train_batch_size * args.gradient_accumulation_steps
            }, step=state.global_step)


# Add NEFTune implementation
class NEFTuneCallback(TrainerCallback):
    """
    Implements NEFTune technique (Noisy Embedding Fine-Tuning)
    Paper: https://arxiv.org/abs/2310.05914
    """
    def __init__(self, noise_alpha=0.05):
        self.noise_alpha = noise_alpha
        
    def on_step_begin(self, args, state, control, **kwargs):
        # Get model from kwargs
        model = kwargs.get("model", None)
        if model is None:
            return
        
        # Check if we're in training mode
        if not model.training:
            return
        
        # Apply noisy embeddings only during training
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            # Store original forward method if we haven't already
            if not hasattr(self, "original_forward"):
                self.original_forward = embeddings.forward
                
                # Define noisy forward method
                def noisy_forward(input_ids, **kwargs):
                    # Get the original embeddings
                    embeds = self.original_forward(input_ids, **kwargs)
                    
                    # Apply noise during training
                    if model.training:
                        # Generate noise with the same shape as embeddings
                        noise = torch.randn_like(embeds) * self.noise_alpha
                        
                        # Add noise to embeddings
                        embeds = embeds + noise
                    
                    return embeds
                
                # Replace the forward method
                embeddings.forward = noisy_forward
    
    def on_train_end(self, args, state, control, **kwargs):
        # Restore original forward method if we modified it
        model = kwargs.get("model", None)
        if model is None:
            return
            
        if hasattr(self, "original_forward") and hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            embeddings.forward = self.original_forward


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2-1B-Instruct for multilingual summarization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name or path (default: meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the HuggingFace dataset directory")
    parser.add_argument("--language", type=str, required=True,
                        help="Language code (e.g., 'en', 'fr', 'de', 'ja')")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for saving model")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Training batch size (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum input sequence length (default: 4096)")
    parser.add_argument("--max_target_length", type=int, default=256,
                        help="Maximum target sequence length (default: 256)")
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
    parser.add_argument("--wandb_project", type=str, default="mulsum-llama",
                        help="Weights & Biases project name (default: mulsum-llama)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (default: None)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (default: None)")
    parser.add_argument("--rouge_eval_samples", type=int, default=10,
                        help="Number of samples to use for ROUGE evaluation (default: 10)")
    parser.add_argument("--no_rouge", action="store_true",
                        help="Disable ROUGE evaluation during training")
    parser.add_argument("--use_all_linear_layers", action="store_true", default=True,
                        help="Apply LoRA to all linear layers in the model (default: True)")
    parser.add_argument("--use_neftune", action="store_true", default=True,
                        help="Enable NEFTune (Noisy Embedding Fine-Tuning) (default: True)")
    parser.add_argument("--neftune_noise_alpha", type=float, default=5.0,
                        help="Noise alpha for NEFTune (default: 5.0)")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of evaluations with no improvement after which training will be stopped (default: 3)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01,
                        help="Minimum change to qualify as an improvement (default: 0.01)")
    parser.add_argument("--use_flash_attention", action="store_true", default=True,
                        help="Enable Flash Attention 2 for faster training (default: True)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Initialize wandb for experiment tracking
    run_name = args.wandb_run_name or f"llama32-{args.language}-r{args.lora_r}-lr{args.learning_rate}"
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
            "prompt_style": "simple_article_summary_format",
            "rouge_evaluation": not args.no_rouge,
            "rouge_eval_samples": args.rouge_eval_samples,
            "use_all_linear_layers": args.use_all_linear_layers,
            "use_neftune": args.use_neftune,
            "neftune_noise_alpha": args.neftune_noise_alpha,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": seed,
            "use_flash_attention": args.use_flash_attention
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
    
    # Setup Flash Attention 2 if enabled and available
    attn_implementation = "flash_attention_2" if args.use_flash_attention else "eager"
    
    # Load tokenizer with special tokens for Llama-3.2
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure the special tokens are properly set
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = tokenizer.eos_token
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model with quantization and set use_cache=False for gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Required for gradient checkpointing
        attn_implementation=attn_implementation
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Step 2: Set up LoRA
    logger.info(f"Setting up LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Determine target modules based on user preference
    if args.use_all_linear_layers:
        # Find all linear layers in the model
        target_modules = find_linear_layers(model)
        logger.info(f"Targeting all linear layers: {target_modules}")
    else:
        # Default target modules for Llama-3.2 architecture
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
            "gate_proj", "up_proj", "down_proj"  # MLP modules
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
    train_dataset_processed = preprocess_dataset(
        train_dataset, 
        tokenizer, 
        language=args.language,
        max_length=args.max_length, 
        max_target_length=args.max_target_length
    )
    
    val_dataset_processed = preprocess_dataset(
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
        logging_steps=50,
        eval_strategy="epoch",  
        save_strategy="epoch",
        save_total_limit=5,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        group_by_length=True,
        fp16=False,
        gradient_checkpointing=True,
        label_names=["labels"],
        deepspeed=None,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        run_name=run_name,
    )
    
    # Step 7: Create trainer with callbacks
    callbacks = [
        WandbMetricsCallback(),
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
    ]
    
    # Add NEFTune callback if enabled
    if args.use_neftune:
        logger.info(f"Enabling NEFTune with noise_alpha={args.neftune_noise_alpha}")
        callbacks.append(NEFTuneCallback(noise_alpha=args.neftune_noise_alpha))
    
    # Add ROUGE evaluation callback if not disabled
    if not args.no_rouge and ROUGE_AVAILABLE:
        logger.info(f"Adding ROUGE evaluation callback with {args.rouge_eval_samples} samples")
        rouge_callback = RougeEvaluationCallback(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=val_dataset_processed,
            raw_dataset=val_dataset,
            language=args.language,
            max_samples=args.rouge_eval_samples
        )
        callbacks.append(rouge_callback)
    elif args.no_rouge:
        logger.info("ROUGE evaluation disabled by user")
    elif not ROUGE_AVAILABLE:
        logger.warning("ROUGE package not available - skipping ROUGE evaluation")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=val_dataset_processed,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Log a sample prompt for reference
    try:
        # Get the prompt format we're using
        sample_prompt = create_summary_prompt("SAMPLE_ARTICLE_TEXT", args.language)
        
        # Log to wandb
        wandb.config.update({
            "sample_prompt_format": sample_prompt
        })
    except Exception as e:
        logger.warning(f"Error logging sample prompt: {e}")
    
    # Step 8: Train and save model
    logger.info("Starting training")
    trainer.train()
    
    # Step 9: Save the model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Generate a final evaluation report on the best model
    if not args.no_rouge and ROUGE_AVAILABLE:
        logger.info("Generating final evaluation report...")
        # Take more samples for final evaluation
        final_samples = min(50, len(val_dataset))
        indices = list(range(len(val_dataset)))
        sample_indices = np.random.choice(indices, final_samples, replace=False)
        
        rouge_scores = []
        
        for idx in tqdm(sample_indices, desc="Evaluating final model"):
            example = val_dataset[int(idx)]
            reference_summary = example["summary"]
            
            generated_summary = generate_model_summary(
                model,
                tokenizer,
                example["text"],
                args.language
            )
            
            if not generated_summary:
                continue
                
            scores = calculate_rouge_scores(reference_summary, generated_summary)
            rouge_scores.append(scores)
        
        if rouge_scores:
            # Calculate average scores
            final_scores = {
                metric: np.mean([score[metric] for score in rouge_scores])
                for metric in ["rouge-1", "rouge-2", "rouge-l"]
            }
            
            # Log final scores
            logger.info("Final evaluation results:")
            logger.info(f"ROUGE-1: {final_scores['rouge-1']:.4f}")
            logger.info(f"ROUGE-2: {final_scores['rouge-2']:.4f}")
            logger.info(f"ROUGE-L: {final_scores['rouge-l']:.4f}")
            
            # Save results to file
            with open(os.path.join(args.output_dir, "final_evaluation.json"), "w") as f:
                json.dump(final_scores, f, indent=2)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    f"rouge/final_{metric}": score
                    for metric, score in final_scores.items()
                })
    
    # Log model size after training
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    wandb.log({"model_size_mb": model_size_mb})

    # Finish wandb run
    wandb.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()