#!/usr/bin/env python3
"""
compare_qwen.py - Qualitatively compare base and fine-tuned Qwen models
for summarization tasks by displaying side-by-side examples.
"""

import os
import argparse
import json
import torch
import random
import re
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_summary_prompt(text: str, language: str) -> str:
    """Create a language-specific prompt for summarization based on original code"""
    # Truncate input if needed (to match input function)
    if len(text) > 4096:
        text = text[:4096]
    
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
        # English for other languages, shouldn't happen becasue we only use fr de en ja ru, but for completeness
        prompt = f"""Please provide a concise summary of the following article in {language}. 
        The summary should be comprehensive, capturing all key points and main arguments, 
        but avoid unnecessary details. Output only the summary.

        Article:
        {text}

        Summary:"""
    
    return prompt


def generate_summary(model, tokenizer, text, language="en", max_input_length=4096, max_new_tokens=256, debug=False):
    """Generate a summary for a given text using the specified model - matching generate_summaries.py approach"""
    # Create prompt using the original format
    prompt = create_summary_prompt(text, language)
    
    if debug:
        logger.info(f"Prompt begins with: {prompt[:100]}...")
        logger.info(f"Prompt ends with: ...{prompt[-50:]}")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    if debug:
        logger.info(f"Input length in tokens: {len(inputs.input_ids[0])}")
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            attention_mask=inputs.attention_mask
        )
    
    # Decode and extract only the summary part (removing the prompt)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = full_output[len(prompt):].strip()
    
    if debug:
        logger.info(f"Full output length: {len(full_output)} chars")
        logger.info(f"Extracted summary begins with: {summary[:100]}..." if summary else "Empty summary!")
    
    return summary


def truncate_text(text, max_length=150):
    """Truncate text to a reasonable display length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def calculate_metrics(reference, generated):
    """Calculate simple similarity metrics between reference and generated summary"""
    from rouge import Rouge
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(generated, reference)[0]
        
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        }


def check_article_quality(text):
    """Check if an article seems like it might cause issues (e.g., filmography lists)"""
    # Check for possible indicators of article structure problems
    # This is simplistic but can help identify problematic cases
    issues = []
    
    # Check if the article starts with a Wikipedia infobox
    if "{{Infobox" in text[:500]:
        issues.append("Article starts with an infobox")
    
    # Check if article has excessive list structure or is primarily a filmography
    film_indicators = ["film)", "cinema)", "* [[", "filmographie", "acteur", "réalisateur"]
    film_indicator_count = sum(text.lower().count(ind) for ind in film_indicators)
    if film_indicator_count > 10:
        issues.append(f"Article may be primarily filmography (indicators: {film_indicator_count})")
    
    # Check for bullet points which could indicate lists
    if text.count('*') > 15:
        issues.append(f"Article has many bullet points ({text.count('*')})")
    
    return issues


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned Qwen models")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--finetuned_model", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the HuggingFace dataset directory")
    parser.add_argument("--language", type=str, required=True,
                        help="Language code (e.g., 'en', 'fr')")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of examples to compare")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save comparison results (JSON)")
    parser.add_argument("--subset", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Dataset subset to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for example selection")
    parser.add_argument("--max_input_length", type=int, default=4096,
                        help="Maximum input sequence length")
    parser.add_argument("--max_output_length", type=int, default=256,
                        help="Maximum output sequence length")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--skip_problematic", action="store_true",
                        help="Skip articles that might cause issues (like filmographies)")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Step 1: Load dataset
    logger.info(f"Loading {args.subset} dataset from {args.dataset_path}/{args.language}")
    dataset = load_from_disk(os.path.join(args.dataset_path, args.language))
    subset = dataset[args.subset]
    logger.info(f"Loaded {len(subset)} examples from {args.subset} set")
    
    # Step 2: Load base model and tokenizer
    logger.info(f"Loading base model: {args.base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Ensure padding token is set correctly for Qwen
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    logger.info(f"Base model loaded. Type: {type(base_model).__name__}")
    
    # Step 3: Load fine-tuned model
    logger.info(f"Loading fine-tuned model from {args.finetuned_model}")
    
    # Check if adapter config exists (PEFT/LoRA)
    adapter_config_path = os.path.join(args.finetuned_model, "adapter_config.json")
    is_peft_model = os.path.exists(adapter_config_path)
    
    if is_peft_model:
        logger.info(f"Found adapter_config.json - loading as PEFT/LoRA model")
        try:
            # Load PEFT configuration
            peft_config = PeftConfig.from_pretrained(args.finetuned_model)
            logger.info(f"PEFT config: {peft_config}")
            
            # Load as PeftModel (adapter on top of base model)
            finetuned_model = PeftModel.from_pretrained(
                base_model, 
                args.finetuned_model
            )
            logger.info("Successfully loaded as PeftModel")
        except Exception as e:
            logger.warning(f"Error loading as PEFT model: {e}")
            is_peft_model = False
    
    if not is_peft_model:
        logger.info("Loading as standard model (not PEFT/LoRA)")
        try:
            finetuned_model = AutoModelForCausalLM.from_pretrained(
                args.finetuned_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info("Successfully loaded as standard model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    # Step 4: Select samples
    all_samples = []
    skipped_count = 0
    
    # Shuffle dataset for better sampling
    shuffled_indices = list(range(len(subset)))
    random.shuffle(shuffled_indices)
    
    for i in shuffled_indices:
        if len(all_samples) >= args.num_samples:
            break
            
        sample = subset[i]
        text = sample["text"]
        
        if args.skip_problematic:
            issues = check_article_quality(text)
            if issues:
                logger.warning(f"Skipping article {i} due to issues: {', '.join(issues)}")
                skipped_count += 1
                continue
        
        all_samples.append((i, sample))
    
    if skipped_count and args.skip_problematic:
        logger.info(f"Skipped {skipped_count} potentially problematic articles")
    
    if len(all_samples) < args.num_samples:
        logger.warning(f"Could only find {len(all_samples)} suitable samples (requested {args.num_samples})")
    
    # Step 5: Generate summaries and compare
    logger.info("Generating summaries for comparison")
    comparisons = []
    
    console = Console(width=120)
    console.print(Panel(
        f"[bold blue]Comparing Base Qwen Model vs. Fine-tuned Qwen Model[/bold blue]\n"
        f"Base Model: [cyan]{args.base_model}[/cyan]\n"
        f"Fine-tuned Model: [cyan]{args.finetuned_model}[/cyan]\n"
        f"Language: [green]{args.language}[/green]"
    ))
    
    try:
        # Try importing Rouge for metrics
        from rouge import Rouge
        calculate_rouge = True
    except ImportError:
        logger.warning("Rouge package not found. ROUGE metrics will not be calculated.")
        calculate_rouge = False
        console.print("[yellow]Note: Install 'rouge' package for ROUGE metrics: pip install rouge[/yellow]\n")
    
    all_metrics = {
        "base_model": {"rouge-1": [], "rouge-2": [], "rouge-l": []},
        "finetuned_model": {"rouge-1": [], "rouge-2": [], "rouge-l": []}
    }
    
    for idx, (sample_idx, sample) in enumerate(tqdm(all_samples, desc="Comparing models")):
        text = sample["text"]
        reference_summary = sample["summary"]
        
        # Display some info about the article
        if args.debug:
            logger.info(f"\nProcessing article {idx+1}/{len(all_samples)} (dataset index: {sample_idx})")
            logger.info(f"Title: {sample.get('title', 'Unknown')}")
            logger.info(f"Article length: {len(text)} characters")
            logger.info(f"Reference summary length: {len(reference_summary)} characters")
        
        # Generate summaries
        try:
            logger.info(f"Generating base model summary...")
            base_summary = generate_summary(
                base_model, 
                base_tokenizer, 
                text, 
                language=args.language,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_output_length,
                debug=args.debug
            )
        except Exception as e:
            logger.error(f"Error generating base model summary: {e}")
            base_summary = "Error generating summary"
        
        try:
            logger.info(f"Generating fine-tuned model summary...")
            finetuned_summary = generate_summary(
                finetuned_model, 
                base_tokenizer, 
                text, 
                language=args.language,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_output_length,
                debug=args.debug
            )
        except Exception as e:
            logger.error(f"Error generating fine-tuned model summary: {e}")
            finetuned_summary = "Error generating summary"
        
        # Calculate metrics if Rouge is available
        if calculate_rouge:
            base_metrics = calculate_metrics(reference_summary, base_summary)
            finetuned_metrics = calculate_metrics(reference_summary, finetuned_summary)
            
            # Accumulate metrics for averaging later
            for metric_name in base_metrics:
                all_metrics["base_model"][metric_name].append(base_metrics[metric_name])
                all_metrics["finetuned_model"][metric_name].append(finetuned_metrics[metric_name])
        else:
            base_metrics = {}
            finetuned_metrics = {}
        
        # Save comparison
        comparison = {
            "index": sample_idx,
            "title": sample.get("title", ""),
            "text": truncate_text(text, 500),  # Save truncated text for display
            # "full_text": text,  # Save full text
            "reference_summary": reference_summary,
            "base_model_summary": base_summary,
            "finetuned_model_summary": finetuned_summary,
            "base_metrics": base_metrics,
            "finetuned_metrics": finetuned_metrics
        }
        comparisons.append(comparison)
        
        # Print comparison
        table = Table(title=f"Example {idx+1} - {sample.get('title', f'Article {sample_idx}')}")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Summary", style="green")
        
        if calculate_rouge:
            table.add_column("ROUGE-1", style="magenta", no_wrap=True)
            table.add_column("ROUGE-2", style="magenta", no_wrap=True)
            table.add_column("ROUGE-L", style="magenta", no_wrap=True)
        
        # Add rows to the table
        table.add_row("Original Text", truncate_text(text, 300))
        
        if calculate_rouge:
            table.add_row(
                "Reference Summary", 
                reference_summary,
                "-", "-", "-"
            )
            table.add_row(
                "Base Model Summary", 
                base_summary,
                f"{base_metrics['rouge-1']:.4f}", 
                f"{base_metrics['rouge-2']:.4f}", 
                f"{base_metrics['rouge-l']:.4f}"
            )
            table.add_row(
                "Fine-tuned Model Summary", 
                finetuned_summary,
                f"{finetuned_metrics['rouge-1']:.4f}", 
                f"{finetuned_metrics['rouge-2']:.4f}", 
                f"{finetuned_metrics['rouge-l']:.4f}"
            )
        else:
            table.add_row("Reference Summary", reference_summary)
            table.add_row("Base Model Summary", base_summary)
            table.add_row("Fine-tuned Model Summary", finetuned_summary)
        
        console.print(table)
        console.print("\n")
    
    # Step 6: Save results if output file is specified
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "base_model": args.base_model,
                    "finetuned_model": args.finetuned_model,
                    "dataset_path": args.dataset_path,
                    "language": args.language,
                    "subset": args.subset,
                    "num_samples": args.num_samples,
                },
                "comparisons": comparisons,
                "average_metrics": {
                    "base_model": {
                        metric: sum(values)/len(values) if values else 0 
                        for metric, values in all_metrics["base_model"].items()
                    },
                    "finetuned_model": {
                        metric: sum(values)/len(values) if values else 0 
                        for metric, values in all_metrics["finetuned_model"].items()
                    }
                } if calculate_rouge else {}
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison results saved to {args.output_file}")
    
    # Step 7: Print average metrics if available
    if calculate_rouge and all_metrics["base_model"]["rouge-1"] and all_metrics["finetuned_model"]["rouge-1"]:
        metrics_table = Table(title="Average ROUGE Metrics")
        metrics_table.add_column("Model", style="cyan")
        metrics_table.add_column("ROUGE-1", style="magenta")
        metrics_table.add_column("ROUGE-2", style="magenta")
        metrics_table.add_column("ROUGE-L", style="magenta")
        
        for model_name, metrics in all_metrics.items():
            avg_rouge1 = sum(metrics["rouge-1"])/len(metrics["rouge-1"]) if metrics["rouge-1"] else 0
            avg_rouge2 = sum(metrics["rouge-2"])/len(metrics["rouge-2"]) if metrics["rouge-2"] else 0
            avg_rougeL = sum(metrics["rouge-l"])/len(metrics["rouge-l"]) if metrics["rouge-l"] else 0
            
            metrics_table.add_row(
                "Base Qwen Model" if model_name == "base_model" else "Fine-tuned Qwen Model",
                f"{avg_rouge1:.4f}",
                f"{avg_rouge2:.4f}",
                f"{avg_rougeL:.4f}"
            )
        
        console.print(metrics_table)
        console.print("\n")
        
        # Calculate improvements
        avg_base_rouge1 = sum(all_metrics["base_model"]["rouge-1"])/len(all_metrics["base_model"]["rouge-1"])
        avg_ft_rouge1 = sum(all_metrics["finetuned_model"]["rouge-1"])/len(all_metrics["finetuned_model"]["rouge-1"])
        
        avg_base_rouge2 = sum(all_metrics["base_model"]["rouge-2"])/len(all_metrics["base_model"]["rouge-2"])
        avg_ft_rouge2 = sum(all_metrics["finetuned_model"]["rouge-2"])/len(all_metrics["finetuned_model"]["rouge-2"])
        
        avg_base_rougeL = sum(all_metrics["base_model"]["rouge-l"])/len(all_metrics["base_model"]["rouge-l"])
        avg_ft_rougeL = sum(all_metrics["finetuned_model"]["rouge-l"])/len(all_metrics["finetuned_model"]["rouge-l"])
        
        improvement1 = ((avg_ft_rouge1 - avg_base_rouge1) / avg_base_rouge1) * 100 if avg_base_rouge1 > 0 else 0
        improvement2 = ((avg_ft_rouge2 - avg_base_rouge2) / avg_base_rouge2) * 100 if avg_base_rouge2 > 0 else 0
        improvementL = ((avg_ft_rougeL - avg_base_rougeL) / avg_base_rougeL) * 100 if avg_base_rougeL > 0 else 0
        
        improvement_msg = f"ROUGE-1 Improvement: {improvement1:.2f}%\n"
        improvement_msg += f"ROUGE-2 Improvement: {improvement2:.2f}%\n"
        improvement_msg += f"ROUGE-L Improvement: {improvementL:.2f}%"
        
        console.print(Panel(improvement_msg, title="Fine-tuning Improvements", border_style="green"))
        console.print("\n")
    
    # Step 8: Print summary of observations
    console.print(Panel(
        "[bold]Comparison Summary[/bold]\n\n"
        "Review the examples above to observe:\n"
        "1. How well the fine-tuned model captures the essential information\n"
        "2. If the fine-tuned model produces more concise or comprehensive summaries\n"
        "3. Whether the fine-tuned model better handles the specific language style or vocabulary\n"
        "4. Any consistent improvements or issues compared to the base model\n\n"
        "When examining the summaries, consider:\n"
        "• Content coverage: Does the summary contain all key points?\n"
        "• Abstraction: Does it reformulate information rather than copy?\n"
        "• Language: Is the summary well-written and fluent?\n"
        "• Length: Is the summary appropriately concise?\n"
        "• Factual accuracy: Does the summary avoid hallucinations?"
    ))


if __name__ == "__main__":
    main()