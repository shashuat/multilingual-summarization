#!/usr/bin/env python3
"""
evaluate_model.py - Evaluate fine-tuned summarization model performance

This script evaluates a fine-tuned summarization model using various metrics
and provides detailed analysis of its performance.
"""

import os
import argparse
from typing import Optional
from transformers import pipeline

# Import custom modules
from src.archive.model_utils import load_finetuned_model
from src.archive.data_utils import load_dataset_split
from src.archive.metrics_utils import (
    generate_summaries,
    compute_metrics,
    analyze_examples,
    plot_evaluation_results,
    save_evaluation_results
)


def evaluate_model(
    model_path: str,
    dataset_path: str,
    language: str,
    base_model: Optional[str] = None,
    is_lora: bool = False,
    output_dir: Optional[str] = None,
    max_input_length: int = 1024,
    max_target_length: int = 256,
    test_subset_size: Optional[int] = None,
):
    """Evaluate a fine-tuned summarization model"""
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path), "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_finetuned_model(model_path, base_model, is_lora)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}/{language}...")
    dataset = load_dataset_split(dataset_path, language)
    test_dataset = dataset["test"]
    
    # Use subset for testing if specified
    if test_subset_size and test_subset_size < len(test_dataset):
        test_dataset = test_dataset.select(range(test_subset_size))
    
    print(f"Evaluating on {len(test_dataset)} test examples")
    
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
    
    # Get reference summaries and source texts
    reference_summaries = test_dataset["summary"]
    source_texts = test_dataset["text"]
    
    # Compute metrics
    print("Computing evaluation metrics...")
    metrics, metrics_data = compute_metrics(
        generated_summaries, 
        reference_summaries,
        lang=language
    )
    
    # Analyze best/worst examples
    print("Analyzing best and worst examples...")
    examples = analyze_examples(
        generated_summaries, 
        reference_summaries, 
        source_texts
    )
    
    # Generate plots
    print("Generating plots...")
    plot_evaluation_results(metrics_data, output_dir)
    
    # Create config info
    config = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "language": language,
        "test_examples": len(test_dataset),
        "is_lora": is_lora,
        "base_model": base_model,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length,
    }
    
    # Save results
    save_evaluation_results(metrics, examples, config, output_dir)
    
    # Save samples file with generated summaries
    samples = []
    for i, (gen, ref, source) in enumerate(zip(generated_summaries, reference_summaries, source_texts)):
        if i >= 50:  # Limit to first 50 examples to keep file manageable
            break
        samples.append({
            "source": source[:1000] + ("..." if len(source) > 1000 else ""),  # Truncate for readability
            "reference": ref,
            "generated": gen,
        })
    
    samples_path = os.path.join(output_dir, "generated_samples.json")
    import json
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # Print metrics
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    print(f"\nGenerated samples saved to {samples_path}")
    
    return metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned summarization model")
    
    parser.add_argument("--model-path", required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--dataset-path", required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--language", required=True,
                        help="Language to evaluate on (e.g., fr, de)")
    parser.add_argument("--base-model", default=None,
                        help="Base model name/path (required for LoRA models)")
    parser.add_argument("--lora", action="store_true",
                        help="Whether the model is a LoRA fine-tuned model")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for evaluation results")
    parser.add_argument("--max-input-length", type=int, default=1024,
                        help="Maximum input length for tokenization")
    parser.add_argument("--max-target-length", type=int, default=256,
                        help="Maximum target length for tokenization")
    parser.add_argument("--test-subset", type=int, default=None,
                        help="Use a subset of test data (for quicker evaluation)")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        language=args.language,
        base_model=args.base_model,
        is_lora=args.lora,
        output_dir=args.output_dir,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        test_subset_size=args.test_subset,
    )


if __name__ == "__main__":
    main()