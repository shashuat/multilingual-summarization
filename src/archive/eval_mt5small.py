#!/usr/bin/env python3
"""
eval_llama.py - Evaluate fine-tuned Llama model on summarization
"""

import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from evaluate import load as load_metric

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prompt(text):
    """Create a prompt for summarization task"""
    return f"""### Instruction:
Summarize the following text in a concise manner.

### Input:
{text}

### Response:
"""


def extract_summary(response):
    """Extract the summary from the model's response"""
    # For Llama-3.2 models, we need to extract just the summary part from the response
    if "### Response:" in response:
        summary = response.split("### Response:")[-1].strip()
    else:
        summary = response.strip()
    
    return summary


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Llama model on summarization")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Base model name for LoRA")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the HuggingFace dataset directory")
    parser.add_argument("--language", type=str, required=True,
                        help="Language code (e.g., 'en', 'fr')")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for evaluation results")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input length")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of examples to evaluate (None for all)")
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model_path), "evaluation")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load model and tokenizer
    logger.info(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Check if pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading base model from {args.base_model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    
    logger.info(f"Loading adapter from {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path)
    
    # Step 2: Load test dataset
    logger.info(f"Loading test dataset from {args.dataset_path}/{args.language}")
    dataset = load_from_disk(os.path.join(args.dataset_path, args.language))
    test_dataset = dataset["test"]
    
    # Use subset if specified
    if args.sample_size and args.sample_size < len(test_dataset):
        logger.info(f"Using {args.sample_size} examples for evaluation")
        test_dataset = test_dataset.select(range(args.sample_size))
    
    logger.info(f"Evaluating on {len(test_dataset)} examples")
    
    # Step 3: Generate summaries
    logger.info("Generating summaries")
    generated_summaries = []
    reference_summaries = test_dataset["summary"]
    
    for i in tqdm(range(0, len(test_dataset), args.batch_size)):
        batch = test_dataset[i:min(i + args.batch_size, len(test_dataset))]
        
        # Create prompts
        prompts = [create_prompt(text) for text in batch["text"]]
        
        # Tokenize inputs
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            padding_side="left"
        ).to(model.device)
        
        # Generate summaries
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        # Decode outputs and extract summaries
        for j, output in enumerate(outputs):
            # Get the new tokens only (exclude input prompt)
            input_length = inputs.input_ids[j].size(0)
            new_tokens = output[input_length:]
            
            # Decode and extract summary
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            summary = extract_summary(decoded)
            generated_summaries.append(summary)
    
    # Step 4: Compute metrics
    logger.info("Computing ROUGE scores")
    rouge = load_metric("rouge")
    rouge_results = rouge.compute(
        predictions=generated_summaries,
        references=reference_summaries,
        use_stemmer=True,
    )

    # breakpoint()
    
    # Format ROUGE scores
    metrics = {
        "rouge1": round(rouge_results["rouge1"], 2),
        "rouge2": round(rouge_results["rouge2"], 2),
        "rougeL": round(rouge_results["rougeL"], 2),
        "rougeLsum": round(rouge_results["rougeLsum"], 2)
    }
    
    # Length statistics
    gen_lengths = [len(summary.split()) for summary in generated_summaries]
    ref_lengths = [len(summary.split()) for summary in reference_summaries]
    
    metrics.update({
        "avg_gen_length": round(np.mean(gen_lengths), 2),
        "avg_ref_length": round(np.mean(ref_lengths), 2),
        "length_ratio": round(np.mean(gen_lengths) / np.mean(ref_lengths), 2),
    })
    
    # Step 5: Save results
    results = {
        "metrics": metrics,
        "examples": [
            {
                "text": test_dataset["text"][i][:500] + "..." if len(test_dataset["text"][i]) > 500 else test_dataset["text"][i],
                "reference": reference_summaries[i],
                "generated": generated_summaries[i],
            }
            for i in range(min(10, len(generated_summaries)))
        ],
        "config": {
            "model_path": args.model_path,
            "base_model": args.base_model,
            "dataset_path": args.dataset_path,
            "language": args.language,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "test_examples": len(test_dataset),
        }
    }
    
    # Save to file
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print metrics
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value}")
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()