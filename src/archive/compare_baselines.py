#!/usr/bin/env python3
"""
compare_baselines.py - Compare fine-tuned model against baselines

This script evaluates multiple models on the same test set to
provide a fair comparison between your fine-tuned model and baselines.
"""

import os
import json
import argparse
import pandas as pd
from typing import Dict, List, Optional
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Import custom modules
from src.archive.model_utils import load_base_model, load_finetuned_model
from src.archive.data_utils import load_dataset_split
from src.archive.metrics_utils import generate_summaries, compute_metrics, plot_model_comparison


def load_model_from_spec(model_spec: Dict) -> tuple:
    """
    Load a model based on its specification from the config file
    
    Args:
        model_spec: Dictionary with model specification
    
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    name = model_spec["name"]
    model_type = model_spec["type"]
    
    print(f"Loading model: {name}")
    
    if model_type == "huggingface":
        # Load standard HuggingFace model
        model, tokenizer = load_base_model(name)
        
    elif model_type == "finetuned":
        # Load standard fine-tuned model
        model, tokenizer = load_finetuned_model(name, is_lora=False)
        
    elif model_type == "lora":
        # Load LoRA model
        base_model = model_spec["base_model"]
        model, tokenizer = load_finetuned_model(name, base_model, is_lora=True)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, tokenizer


def save_example_comparisons(
    test_dataset, 
    all_summaries: Dict[str, List[str]], 
    output_dir: str, 
    language: str, 
    n_examples: int = 10
):
    """
    Save examples of summaries from different models for comparison
    
    Args:
        test_dataset: Test dataset with source texts and reference summaries
        all_summaries: Dictionary of generated summaries by model name
        output_dir: Output directory to save examples
        language: Language code for file naming
        n_examples: Number of examples to save
    """
    examples = []
    
    # Select random examples
    import numpy as np
    indices = np.random.choice(len(test_dataset), min(n_examples, len(test_dataset)), replace=False)
    
    for idx in indices:
        example = {
            "source": test_dataset[idx]["text"][:1000] + "...",  # Truncate for readability
            "reference": test_dataset[idx]["summary"],
        }
        
        # Add summaries from each model
        for model_name, summaries in all_summaries.items():
            example[f"summary_{model_name}"] = summaries[idx]
        
        examples.append(example)
    
    # Save examples
    examples_path = os.path.join(output_dir, f"{language}_summary_examples.json")
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Comparison examples saved to {examples_path}")


def compare_models(
    models_config: List[Dict],
    dataset_path: str,
    language: str,
    output_dir: str,
    max_input_length: int = 1024,
    max_target_length: int = 256,
    test_subset_size: Optional[int] = None,
):
    """Compare multiple models on the same test set"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}/{language}...")
    dataset = load_dataset_split(dataset_path, language)
    test_dataset = dataset["test"]
    
    # Use subset for testing if specified
    if test_subset_size and test_subset_size < len(test_dataset):
        test_dataset = test_dataset.select(range(test_subset_size))
    
    print(f"Evaluating on {len(test_dataset)} test examples")
    
    # Get reference summaries
    reference_summaries = test_dataset["summary"]
    
    # Evaluate each model
    all_metrics = []
    all_summaries = {}
    
    for model_spec in models_config:
        model_name = model_spec["display_name"]
        print(f"\nEvaluating model: {model_name}")
        
        # Load model
        model, tokenizer = load_model_from_spec(model_spec)
        
        # Create summarization pipeline
        device = 0 if model.device.type == "cuda" else -1
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_target_length,
            min_length=min(32, max_target_length // 4),
            do_sample=False,
        )
        
        # Generate summaries
        generated_summaries = generate_summaries(
            summarizer, 
            test_dataset,
            max_input_length=max_input_length
        )
        
        # Save generated summaries
        all_summaries[model_name] = generated_summaries
        
        # Compute metrics
        metrics, _ = compute_metrics(
            generated_summaries, 
            reference_summaries,
            lang=language
        )
        
        # Add model name to metrics
        metrics["model"] = model_name
        all_metrics.append(metrics)
        
        # Print metrics
        print(f"Results for {model_name}:")
        for metric_name, value in metrics.items():
            if metric_name != "model":
                print(f"  {metric_name}: {value}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = os.path.join(output_dir, f"{language}_model_comparison.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Create visualization
    plot_model_comparison(metrics_df, output_dir, language)
    
    # Save example comparisons
    save_example_comparisons(test_dataset, all_summaries, output_dir, language)
    
    print(f"\nComparison results saved to {output_dir}")
    print(f"Metrics saved to {metrics_csv_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare model performance against baselines")
    
    parser.add_argument("--config", required=True,
                        help="Path to models configuration JSON file")
    parser.add_argument("--dataset-path", required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--language", required=True,
                        help="Language to evaluate on (e.g., fr, de)")
    parser.add_argument("--output-dir", default="./model_comparison",
                        help="Output directory for comparison results")
    parser.add_argument("--max-input-length", type=int, default=1024,
                        help="Maximum input length for tokenization")
    parser.add_argument("--max-target-length", type=int, default=256,
                        help="Maximum target length for tokenization")
    parser.add_argument("--test-subset", type=int, default=None,
                        help="Use a subset of test data (for quicker evaluation)")
    
    args = parser.parse_args()
    
    # Load models configuration
    with open(args.config, "r") as f:
        config_data = json.load(f)
        # Support both formats: {"models": [...]} and direct list [...]
        models_config = config_data.get("models", config_data)
    
    compare_models(
        models_config=models_config,
        dataset_path=args.dataset_path,
        language=args.language,
        output_dir=args.output_dir,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        test_subset_size=args.test_subset,
    )


if __name__ == "__main__":
    main()