#!/usr/bin/env python3
"""
compare_llama.py - Qualitatively compare base and fine-tuned Llama models
for summarization tasks by displaying side-by-side examples.
"""

import os
import argparse
import json
import torch
import random
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

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
    # For Llama-3.2 models, extract just the summary part from the response
    if "### Response:" in response:
        summary = response.split("### Response:")[-1].strip()
    else:
        summary = response.strip()
    
    return summary


def generate_summary(model, tokenizer, text, max_input_length=1024, max_new_tokens=256):
    """Generate a summary for a given text using the specified model"""
    # Create prompt
    prompt = create_prompt(text)
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode output and extract summary
    input_length = inputs.input_ids[0].size(0)
    new_tokens = outputs[0][input_length:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return extract_summary(decoded)


def truncate_text(text, max_length=150):
    """Truncate text to a reasonable display length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned Llama models")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
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
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Step 1: Load dataset
    logger.info(f"Loading {args.subset} dataset from {args.dataset_path}/{args.language}")
    dataset = load_from_disk(os.path.join(args.dataset_path, args.language))
    subset = dataset[args.subset]
    
    # Select random samples if the dataset is larger than num_samples
    if len(subset) > args.num_samples:
        indices = random.sample(range(len(subset)), args.num_samples)
        samples = [subset[i] for i in indices]
    else:
        samples = [subset[i] for i in range(len(subset))]
    
    # Step 2: Load base model and tokenizer
    logger.info(f"Loading base model: {args.base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Check if pad token is set
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Step 3: Load fine-tuned model
    logger.info(f"Loading fine-tuned model from {args.finetuned_model}")
    finetuned_model = PeftModel.from_pretrained(
        base_model, 
        args.finetuned_model
    )
    
    # Step 4: Generate summaries and compare
    logger.info("Generating summaries for comparison")
    comparisons = []
    
    console = Console(width=120)
    console.print(Panel(
        f"[bold blue]Comparing Base Model vs. Fine-tuned Model[/bold blue]\n"
        f"Base Model: [cyan]{args.base_model}[/cyan]\n"
        f"Fine-tuned Model: [cyan]{args.finetuned_model}[/cyan]\n"
        f"Language: [green]{args.language}[/green]"
    ))
    
    for i, sample in enumerate(tqdm(samples, desc="Comparing models")):
        text = sample["text"]
        reference_summary = sample["summary"]
        
        # Generate summaries
        base_summary = generate_summary(base_model, base_tokenizer, text)
        finetuned_summary = generate_summary(finetuned_model, base_tokenizer, text)
        
        # Save comparison
        comparison = {
            "index": i + 1,
            "text": truncate_text(text, 500),  # Save truncated text for display
            "full_text": text,  # Save full text
            "reference_summary": reference_summary,
            "base_model_summary": base_summary,
            "finetuned_model_summary": finetuned_summary,
        }
        comparisons.append(comparison)
        
        # Print comparison
        table = Table(title=f"Example {i+1}")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Summary", style="green")
        
        table.add_row("Original Text", truncate_text(text, 300))
        table.add_row("Reference Summary", reference_summary)
        table.add_row("Base Model Summary", base_summary)
        table.add_row("Fine-tuned Model Summary", finetuned_summary)
        
        console.print(table)
        console.print("\n")
    
    # Step 5: Save results if output file is specified
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
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison results saved to {args.output_file}")
    
    # Step 6: Print summary of observations
    console.print(Panel(
        "[bold]Comparison Summary[/bold]\n\n"
        "Review the examples above to observe:\n"
        "1. How well the fine-tuned model captures the essential information\n"
        "2. If the fine-tuned model produces more concise or comprehensive summaries\n"
        "3. Whether the fine-tuned model better handles the specific language style or vocabulary\n"
        "4. Any consistent improvements or issues compared to the base model"
    ))


if __name__ == "__main__":
    main()