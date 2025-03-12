#!/usr/bin/env python3
"""
src/finetune_qwen.py - Full parameter fine-tuning for Qwen2.5-0.5B-Instruct for multilingual summarization
with proper logging, evaluation, and checkpoint saving at each epoch
"""

import os
import argparse
import torch
import wandb
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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

# Import rouge for metrics calculation
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    logger.warning("Rouge package not found. Install with 'pip install rouge' for ROUGE metrics.")
    ROUGE_AVAILABLE = False


def create_summary_prompt(text: str, language: str, summary: str = "") -> str:
    """
    Create a language-specific prompt for summarization
    If summary is provided, it's appended for training examples
    """
    # Truncate input if needed
    max_input_length = 4096
    if len(text) > max_input_length:
        text = text[:max_input_length]
    
    # Create appropriate prompt based on language
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
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Calculate ROUGE scores at the end of each epoch"""
        if not ROUGE_AVAILABLE:
            logger.warning("Rouge package not available. Skipping ROUGE evaluation.")
            return
        
        logger.info(f"Calculating ROUGE metrics at end of epoch {state.epoch}...")
        
        # Take a sample of validation examples
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
            # Convert to Python int
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
            if len(examples) < 3:  # Limit to 3 examples for cleaner logs
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
            
            # Log epoch number explicitly
            wandb.log({"epoch": state.epoch}, step=state.global_step)
            
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
        logger.info(f"Epoch {state.epoch} - ROUGE-1: {avg_scores['rouge-1']:.4f}, ROUGE-2: {avg_scores['rouge-2']:.4f}, ROUGE-L: {avg_scores['rouge-l']:.4f}")
        
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


# Custom callback to save a checkpoint at the end of each epoch
class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, save_path, tokenizer, model):
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.model = model
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Save model checkpoint at the end of each epoch"""
        epoch = round(state.epoch)  # Get current epoch number (rounded to handle fractional epochs)
        
        # Create epoch-specific directory
        epoch_dir = os.path.join(self.save_path, f"epoch-{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        logger.info(f"Saving checkpoint for epoch {epoch} to {epoch_dir}")
        
        # Save model and tokenizer
        self.model.save_pretrained(epoch_dir)
        self.tokenizer.save_pretrained(epoch_dir)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "checkpoint/epoch": epoch,
                "checkpoint/path": epoch_dir
            }, step=state.global_step)
        
        logger.info(f"Checkpoint saved for epoch {epoch}")


# Custom callback to log detailed metrics to W&B
class WandbMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and wandb.run is not None:
            # Add epoch number to metrics
            metrics["epoch"] = round(state.epoch)
            
            # Log detailed metrics
            wandb.log(metrics, step=state.global_step)
            
            # Log example count and learning rate
            lr = args.learning_rate
            if hasattr(args, "get_last_lr") and callable(args.get_last_lr):
                lr = args.get_last_lr()[0]
                
            wandb.log({
                "train/example_count": state.global_step * args.train_batch_size * args.gradient_accumulation_steps,
                "train/learning_rate": lr
            }, step=state.global_step)
            
            logger.info(f"Evaluation at epoch {round(state.epoch)} - Loss: {metrics.get('eval_loss', 'N/A')}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Full parameter fine-tuning of Qwen2.5-0.5B-Instruct for multilingual summarization")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model name or path (default: Qwen/Qwen2.5-0.5B-Instruct)")
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
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum input sequence length (default: 4096)")
    parser.add_argument("--max_target_length", type=int, default=256,
                        help="Maximum target sequence length (default: 256)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer (default: 0.01)")
    parser.add_argument("--wandb_project", type=str, default="mulsum-qwen",
                        help="Weights & Biases project name (default: mulsum-qwen)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (default: None)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (default: None)")
    parser.add_argument("--rouge_eval_samples", type=int, default=10,
                        help="Number of samples to use for ROUGE evaluation (default: 10)")
    parser.add_argument("--no_rouge", action="store_true",
                        help="Disable ROUGE evaluation during training")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed precision training (fp16)")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable mixed precision training (bf16)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb for experiment tracking
    run_name = args.wandb_run_name or f"qwen-{args.language}-full-finetune-ep{args.num_epochs}"
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
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "prompt_style": "simple_article_summary_format",
            "rouge_evaluation": not args.no_rouge,
            "rouge_eval_samples": args.rouge_eval_samples,
            "finetune_method": "full_parameter_finetune",
            "precision": "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32")
        }
    )
    
    # Step 1: Load model and tokenizer
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    
    # Choose precision based on args
    compute_dtype = None
    if args.fp16:
        compute_dtype = torch.float16
        logger.info("Using fp16 precision")
    elif args.bf16:
        compute_dtype = torch.bfloat16
        logger.info("Using bf16 precision")
    else:
        logger.info("Using fp32 precision (full precision training)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure the special tokens are properly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Load model with appropriate configuration
    load_kwargs = {
        "device_map": "auto",
        "use_cache": False  # Disable KV cache for training
    }
    
    if compute_dtype is not None:
        load_kwargs["torch_dtype"] = compute_dtype
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    
    # Log trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
    wandb.config.update({
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": 100 * trainable_params / all_params
    })
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled")
    
    # Step 2: Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}/{args.language}")
    try:
        dataset = load_from_disk(os.path.join(args.dataset_path, args.language))
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        
        logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    except Exception as e:
        # Try alternative path structure if the first attempt fails
        logger.warning(f"Error loading dataset from {args.dataset_path}/{args.language}: {e}")
        logger.info(f"Trying to load dataset directly from {args.dataset_path}")
        
        dataset = load_from_disk(args.dataset_path)
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        
        logger.info(f"Successfully loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    wandb.config.update({
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset)
    })
    
    # Step 3: Preprocess dataset
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
    
    # Step 4: Create data collator that handles padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    
    # Step 5: Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        evaluation_strategy="epoch",
        eval_steps=1,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=args.num_epochs,  # Save a checkpoint for each epoch
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
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        label_names=["labels"],
        optim="adamw_torch_fused",
        run_name=run_name,
        # Set dataloader workers based on system capabilities
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        dataloader_pin_memory=True,
    )
    
    # Step 6: Create trainer with callbacks
    callbacks = [WandbMetricsCallback()]
    
    # Add ROUGE evaluation callback
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
    
    # Add checkpoint saving callback for each epoch
    save_callback = SaveCheckpointCallback(
        save_path=args.output_dir,
        tokenizer=tokenizer,
        model=model
    )
    callbacks.append(save_callback)

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
        sample_prompt = create_summary_prompt("SAMPLE_ARTICLE_TEXT", args.language)
        wandb.config.update({
            "sample_prompt_format": sample_prompt
        })
    except Exception as e:
        logger.warning(f"Error logging sample prompt: {e}")
    
    # Step 7: Train the model
    logger.info(f"Starting full parameter fine-tuning for {args.num_epochs} epochs")
    train_result = trainer.train()
    
    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Step 8: Save the final model
    logger.info(f"Saving final model to {args.output_dir}")
    model.save_pretrained(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    
    # Log final evaluation
    final_metrics = trainer.evaluate()
    trainer.log_metrics("eval", final_metrics)
    trainer.save_metrics("eval", final_metrics)
    
    # Log model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    wandb.log({"model/size_mb": model_size_mb})
    logger.info(f"Model size: {model_size_mb:.2f} MB")

    # Finish wandb run
    wandb.finish()
    
    logger.info("Training complete! Model saved to output directory.")


if __name__ == "__main__":
    main()