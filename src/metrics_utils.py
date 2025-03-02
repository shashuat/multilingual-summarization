#!/usr/bin/env python3
"""
metrics_utils.py - Utilities for computing metrics and evaluations

This module provides functions for computing various metrics for summarization
models and analyzing the results.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import load as load_metric
from transformers import PreTrainedTokenizer, Pipeline
from datasets import Dataset


def compute_rouge_for_trainer(eval_preds, tokenizer, rouge_metric=None):
    """
    Compute ROUGE metrics for HuggingFace Trainer evaluation
    
    Args:
        eval_preds: Tuple of predictions and labels from Trainer
        tokenizer: Tokenizer to decode the predictions and labels
        rouge_metric: Optional pre-loaded ROUGE metric
    
    Returns:
        metrics: Dictionary of ROUGE metrics
    """
    if rouge_metric is None:
        rouge_metric = load_metric("rouge")
    
    preds, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 with tokenizer.pad_token_id to decode properly
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects newlines after each sentence
    decoded_preds = ["\n".join(pred.split(". ")) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.split(". ")) for label in decoded_labels]
    
    # Compute ROUGE scores
    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Extract ROUGE scores
    result = {k: round(v * 100, 2) for k, v in result.items()}
    
    # Add mean generated length
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result


def generate_summaries(
    summarizer: Pipeline,
    dataset: Dataset,
    max_input_length: int = 1024,
    batch_size: int = 8
) -> List[str]:
    """
    Generate summaries for a dataset using a summarization pipeline
    
    Args:
        summarizer: HuggingFace pipeline for summarization
        dataset: Dataset with texts to summarize
        max_input_length: Maximum input length for the model
        batch_size: Batch size for processing
    
    Returns:
        summaries: List of generated summaries
    """
    generated_summaries = []
    
    # Process dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating summaries"):
        batch = dataset[i:min(i + batch_size, len(dataset))]
        
        # Truncate inputs if needed
        texts = [text[:max_input_length] for text in batch["text"]]
        
        # Generate summaries
        summaries = summarizer(texts, truncation=True)
        
        # Extract generated text
        for summary in summaries:
            generated_summaries.append(summary["summary_text"])
    
    return generated_summaries


def compute_metrics(
    generated_summaries: List[str],
    reference_summaries: List[str],
    lang: str = "en"
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute various evaluation metrics for generated summaries
    
    Args:
        generated_summaries: List of generated summaries
        reference_summaries: List of reference summaries
        lang: Language code for BERTScore
    
    Returns:
        metrics: Dictionary of metrics
        metrics_data: Additional data for analysis
    """
    # Load metrics
    rouge = load_metric("rouge")
    bertscore = load_metric("bertscore")
    
    # Compute ROUGE scores
    rouge_results = rouge.compute(
        predictions=generated_summaries,
        references=reference_summaries,
        use_stemmer=True
    )
    
    # Convert to more readable format
    rouge_dict = {
        "rouge1": round(rouge_results["rouge1"].mid.fmeasure * 100, 2),
        "rouge2": round(rouge_results["rouge2"].mid.fmeasure * 100, 2),
        "rougeL": round(rouge_results["rougeL"].mid.fmeasure * 100, 2),
        "rougeLsum": round(rouge_results["rougeLsum"].mid.fmeasure * 100, 2),
        "rouge1_precision": round(rouge_results["rouge1"].mid.precision * 100, 2),
        "rouge1_recall": round(rouge_results["rouge1"].mid.recall * 100, 2),
        "rouge2_precision": round(rouge_results["rouge2"].mid.precision * 100, 2),
        "rouge2_recall": round(rouge_results["rouge2"].mid.recall * 100, 2),
    }
    
    # Compute BERTScore
    print(f"Computing BERTScore (this may take a while)...")
    bertscore_results = bertscore.compute(
        predictions=generated_summaries,
        references=reference_summaries,
        lang=lang,
        batch_size=8
    )
    
    # Calculate average scores
    bertscore_dict = {
        "bertscore_precision": round(np.mean(bertscore_results["precision"]) * 100, 2),
        "bertscore_recall": round(np.mean(bertscore_results["recall"]) * 100, 2),
        "bertscore_f1": round(np.mean(bertscore_results["f1"]) * 100, 2),
    }
    
    # Calculate length statistics
    summary_lengths = [len(summary.split()) for summary in generated_summaries]
    reference_lengths = [len(summary.split()) for summary in reference_summaries]
    
    length_stats = {
        "avg_gen_length": round(np.mean(summary_lengths), 2),
        "avg_ref_length": round(np.mean(reference_lengths), 2),
        "median_gen_length": round(np.median(summary_lengths), 2),
        "median_ref_length": round(np.median(reference_lengths), 2),
        "length_ratio": round(np.mean(summary_lengths) / np.mean(reference_lengths), 2),
    }
    
    # Calculate length correlation
    length_corr, _ = pearsonr(summary_lengths, reference_lengths)
    length_stats["length_correlation"] = round(length_corr, 3)
    
    # Combine all metrics
    metrics = {
        **rouge_dict,
        **bertscore_dict,
        **length_stats,
    }
    
    # Additional data for analysis
    metrics_data = {
        "generated_lengths": summary_lengths,
        "reference_lengths": reference_lengths,
        "rouge_scores": [rouge_results["rouge1"].mid.fmeasure * 100 for _ in range(len(generated_summaries))],
        "bertscore_f1": bertscore_results["f1"]
    }
    
    return metrics, metrics_data


def analyze_examples(
    generated_summaries: List[str],
    reference_summaries: List[str],
    source_texts: List[str],
    n_examples: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze best and worst performing examples based on ROUGE scores
    
    Args:
        generated_summaries: List of generated summaries
        reference_summaries: List of reference summaries
        source_texts: List of source texts
        n_examples: Number of examples to include in each category
    
    Returns:
        examples: Dictionary of best and worst examples
    """
    # Compute per-example ROUGE scores
    rouge = load_metric("rouge")
    per_example_rouge = []
    
    for gen, ref in zip(generated_summaries, reference_summaries):
        score = rouge.compute(
            predictions=[gen],
            references=[ref],
            use_stemmer=True
        )
        per_example_rouge.append(score["rouge1"].mid.fmeasure)
    
    # Find best and worst examples
    indices = np.argsort(per_example_rouge)
    worst_indices = indices[:n_examples]
    best_indices = indices[-n_examples:]
    
    # Prepare examples
    examples = {
        "best": [{
            "source": source_texts[i],
            "reference": reference_summaries[i],
            "generated": generated_summaries[i],
            "rouge1": per_example_rouge[i] * 100,
        } for i in best_indices],
        "worst": [{
            "source": source_texts[i],
            "reference": reference_summaries[i],
            "generated": generated_summaries[i],
            "rouge1": per_example_rouge[i] * 100,
        } for i in worst_indices],
    }
    
    return examples


def plot_evaluation_results(metrics_data: Dict[str, Any], output_dir: str) -> None:
    """
    Create visualizations of evaluation results
    
    Args:
        metrics_data: Dictionary of metrics data
        output_dir: Directory to save the plots
    """
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))
    
    # Create output directory
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Summary length distribution
    plt.subplot(2, 2, 1)
    sns.histplot(
        data={
            "Generated": metrics_data["generated_lengths"],
            "Reference": metrics_data["reference_lengths"]
        },
        bins=20,
        alpha=0.6
    )
    plt.title("Summary Length Distribution")
    plt.xlabel("Length (words)")
    plt.ylabel("Frequency")
    
    # 2. Length correlation
    plt.subplot(2, 2, 2)
    sns.scatterplot(
        x=metrics_data["reference_lengths"],
        y=metrics_data["generated_lengths"],
        alpha=0.5
    )
    plt.title("Summary Length Correlation")
    plt.xlabel("Reference Length (words)")
    plt.ylabel("Generated Length (words)")
    
    # 3. Rouge scores vs length
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        x=metrics_data["generated_lengths"],
        y=metrics_data["rouge_scores"],
        alpha=0.5
    )
    plt.title("ROUGE vs Generated Length")
    plt.xlabel("Generated Length (words)")
    plt.ylabel("ROUGE-1 Score")
    
    # 4. BERTScore distribution
    if "bertscore_f1" in metrics_data:
        plt.subplot(2, 2, 4)
        sns.histplot(
            data=pd.DataFrame({"BERTScore F1": metrics_data["bertscore_f1"]}),
            x="BERTScore F1",
            bins=20
        )
        plt.title("BERTScore F1 Distribution")
        plt.xlabel("BERTScore F1")
        plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "summary_analysis.png"), dpi=300)
    plt.close()
    
    print(f"Plots saved to {plot_dir}")


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str,
    language: str
) -> None:
    """
    Create comparison visualizations for multiple models
    
    Args:
        metrics_df: DataFrame with metrics for different models
        output_dir: Directory to save the plots
        language: Language code for file naming
    """
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 10))
    
    # 1. ROUGE scores comparison
    plt.subplot(2, 2, 1)
    rouge_df = pd.melt(
        metrics_df,
        id_vars=["model"],
        value_vars=["rouge1", "rouge2", "rougeL"],
        var_name="Metric",
        value_name="Score"
    )
    sns.barplot(x="model", y="Score", hue="Metric", data=rouge_df)
    plt.title("ROUGE Scores Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Score")
    plt.legend(title="Metric")
    
    # 2. BERTScore comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x="model", y="bertscore_f1", data=metrics_df)
    plt.title("BERTScore F1 Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("BERTScore F1")
    
    # 3. Length comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x="model", y="avg_gen_length", data=metrics_df)
    plt.title("Average Generated Summary Length")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Words")
    
    # 4. Length ratio comparison
    plt.subplot(2, 2, 4)
    sns.barplot(x="model", y="length_ratio", data=metrics_df)
    plt.title("Length Ratio (Generated/Reference)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Ratio")
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{language}_model_comparison.png"), dpi=300)
    plt.close()


def save_evaluation_results(
    metrics: Dict[str, float],
    examples: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Save evaluation results to disk
    
    Args:
        metrics: Dictionary of metrics
        examples: Dictionary of example analyses
        config: Configuration used for evaluation
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results structure
    results = {
        "metrics": metrics,
        "examples": examples,
        "config": config
    }
    
    # Save full results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save metrics as CSV for easier analysis
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    print(f"Evaluation results saved to {results_path}")
    print(f"Metrics saved to {metrics_csv_path}")