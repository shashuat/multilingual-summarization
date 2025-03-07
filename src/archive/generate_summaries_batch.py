#!/usr/bin/env python3
"""
generate_summaries.py - Generate summaries for multilingual Wikipedia articles using LLMs
with optimized batch processing for faster generation

This script loads raw Wikipedia articles, generates summaries using an LLM,
and saves the summaries to disk for later use in dataset creation.
"""

import os
import json
import argparse
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from time import time


def load_article_index(raw_data_dir: str) -> List[Dict[str, Any]]:
    """Load the article index file containing metadata about all articles"""
    index_path = os.path.join(raw_data_dir, "article_index.json")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Article index not found at {index_path}")
    
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_article_content(raw_data_dir: str, article_id: str, language: str) -> Optional[Dict[str, Any]]:
    """Load content for a specific article in a specific language"""
    lang_dir = os.path.join(raw_data_dir, language)
    article_path = os.path.join(lang_dir, f"{article_id}.json")
    
    if not os.path.exists(article_path):
        return None
            
    try:
        with open(article_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            return {
                "article_id": article_id,
                "language": language,
                "title": content.get("title", ""),
                "text": content.get("text", ""),
                "url": content.get("url", "")
            }
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {article_path}")
        return None


def setup_llm(model_name: str, precision: str = "4bit", gpu_memory_usage: float = 0.7):
    """
    Set up the LLM for summary generation with appropriate quantization
    
    Args:
        model_name: Name of the model to load
        precision: Precision to use (4bit, 8bit, 16bit, or 32bit)
        gpu_memory_usage: Fraction of GPU memory to use (0-1)
    """
    print(f"Loading {model_name} for summary generation...")
    start_time = time()
    
    # Configure quantization based on precision
    quantization_config = None
    dtype = torch.float32
    
    if torch.cuda.is_available():
        if precision == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            dtype = torch.bfloat16
        elif precision == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            dtype = torch.float16
        elif precision == "16bit":
            dtype = torch.float16
        
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set device_map based on GPU availability and memory usage
    if torch.cuda.is_available():
        device_map = {"": 0}  # Use only the first GPU
        # For multi-GPU setup, you could use "auto" instead
    else:
        device_map = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        max_memory={0: f"{int(torch.cuda.get_device_properties(0).total_memory * gpu_memory_usage / 1024**3)}GiB"} if torch.cuda.is_available() else None,
    )
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Print memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU memory used: {memory_used:.2f} GB")
    
    print(f"Model loaded in {time() - start_time:.2f} seconds")
    return model, tokenizer


def create_prompt(text: str, language: str) -> str:
    """Create a prompt for summary generation"""
    return f"""Please provide a concise summary of the following article in {language}. 
The summary should be comprehensive, capturing all key points and main arguments, 
but avoid unnecessary details.

Article:
{text}

Summary:"""


def generate_summaries_batch(
    model, 
    tokenizer, 
    texts: List[str], 
    prompts: List[str],
    max_summary_length: int = 512
) -> List[str]:
    """
    Generate summaries for a batch of articles
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of article texts
        prompts: List of prompts
        max_summary_length: Maximum summary length to generate
        
    Returns:
        List of generated summaries
    """
    # Tokenize all inputs
    encodings = tokenizer(
        prompts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=4096
    ).to(model.device)
    
    # Generate summaries
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask,
            max_new_tokens=max_summary_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
    
    # Decode and extract summaries
    summaries = []
    for i, output in enumerate(outputs):
        full_output = tokenizer.decode(output, skip_special_tokens=True)
        # Extract just the summary part (after the prompt)
        prompt_length = len(prompts[i])
        summary = full_output[prompt_length:].strip()
        summaries.append(summary)
    
    return summaries


def save_summaries(summaries_dir: str, summaries_data: List[Dict[str, Any]]):
    """Save generated summaries to disk"""
    for data in summaries_data:
        lang = data["language"]
        article_id = data["article_id"]
        
        # Ensure language directory exists
        lang_dir = os.path.join(summaries_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        
        # Save summary to file
        summary_path = os.path.join(lang_dir, f"{article_id}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def update_tracking_file(lang_dir: str, completed_articles: set):
    """Update the tracking file with completed articles"""
    tracking_file = os.path.join(lang_dir, "_completed.json")
    with open(tracking_file, "w", encoding="utf-8") as f:
        json.dump({
            "completed_article_ids": list(completed_articles),
            "total_completed": len(completed_articles)
        }, f, ensure_ascii=False, indent=2)


def generate_and_save_summaries(
    raw_data_dir: str,
    summaries_dir: str,
    languages: List[str],
    model_name: str,
    max_input_length: int = 4096,
    max_summary_length: int = 512,
    batch_size: int = 8,  # Increased default batch size
    max_articles: Optional[int] = None,
    resume: bool = True,
    precision: str = "4bit",
    gpu_memory_usage: float = 0.7,
    save_interval: int = 10
) -> None:
    """
    Generate summaries for articles and save them to disk
    
    Args:
        raw_data_dir: Directory containing raw articles
        summaries_dir: Directory to save summaries
        languages: List of languages to process
        model_name: Name of the model to use
        max_input_length: Maximum input length
        max_summary_length: Maximum summary length
        batch_size: Batch size for processing
        max_articles: Maximum number of articles to process
        resume: Whether to resume from previous run
        precision: Model precision (4bit, 8bit, 16bit, 32bit)
        gpu_memory_usage: Fraction of GPU memory to use (0-1)
        save_interval: How often to update tracking file
    """
    # Create output directory for summaries
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Load article index
    print(f"Loading article index from {raw_data_dir}...")
    articles = load_article_index(raw_data_dir)
    
    if max_articles:
        articles = articles[:max_articles]
        
    print(f"Found {len(articles)} articles with parallel content")
    
    # Set up LLM for summary generation
    model, tokenizer = setup_llm(model_name, precision, gpu_memory_usage)
    
    # Dynamic batch size adjustment based on available memory
    if torch.cuda.is_available() and batch_size > 1:
        # Try to estimate the maximum possible batch size
        try:
            # Create a sample prompt and encode
            sample_text = "Sample text for batch size estimation." * 100  # Create moderate length text
            sample_prompt = create_prompt(sample_text, languages[0])
            sample_encoding = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
            
            # Generate with small batch to measure memory usage
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model.generate(
                    sample_encoding.input_ids, 
                    max_new_tokens=max_summary_length // 4,  # Smaller for estimation
                    num_return_sequences=1
                )
            
            memory_per_sample = torch.cuda.max_memory_allocated() * 1.2  # Add 20% buffer
            total_memory = torch.cuda.get_device_properties(0).total_memory * gpu_memory_usage
            estimated_batch_size = max(1, int(total_memory / memory_per_sample))
            
            # Cap at provided batch size
            adjusted_batch_size = min(batch_size, estimated_batch_size)
            
            if adjusted_batch_size < batch_size:
                print(f"Adjusting batch size from {batch_size} to {adjusted_batch_size} based on memory usage")
                batch_size = adjusted_batch_size
                
        except Exception as e:
            print(f"Error estimating batch size: {e}. Using provided batch size {batch_size}.")
    
    # Process each language
    for lang in languages:
        print(f"\nProcessing {lang} articles...")
        
        # Create language directory for summaries
        lang_summaries_dir = os.path.join(summaries_dir, lang)
        os.makedirs(lang_summaries_dir, exist_ok=True)
        
        # Create tracking file for completed summaries
        tracking_file = os.path.join(lang_summaries_dir, "_completed.json")
        completed_articles = set()
        
        # Load tracking file if it exists and resume is enabled
        if resume and os.path.exists(tracking_file):
            with open(tracking_file, "r", encoding="utf-8") as f:
                completed_data = json.load(f)
                completed_articles = set(completed_data.get("completed_article_ids", []))
            print(f"Resuming from {len(completed_articles)} previously completed articles")
        
        # Process articles in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            
            # Prepare batch data
            batch_texts = []
            batch_prompts = []
            batch_ids = []
            batch_titles = []
            skipped = 0
            
            # Collect batch data
            for article in batch:
                article_id = article["id"]
                
                # Skip if already completed
                if article_id in completed_articles:
                    skipped += 1
                    continue
                
                # Load article content
                article_data = load_article_content(raw_data_dir, article_id, lang)
                
                if not article_data or not article_data["text"].strip():
                    print(f"Skipping article {article_id}: Missing {lang} content")
                    skipped += 1
                    continue
                
                # Add to batch
                text = article_data["text"][:max_input_length]
                prompt = create_prompt(text, lang)
                
                batch_texts.append(text)
                batch_prompts.append(prompt)
                batch_ids.append(article_id)
                batch_titles.append(article_data["title"])
            
            # Skip if all articles in batch were already processed
            if len(batch_texts) == 0:
                if skipped > 0:
                    print(f"Skipped {skipped} articles (already processed)")
                continue
            
            # Process batch
            print(f"Generating summaries for batch {i//batch_size + 1}/{len(articles)//batch_size + 1} ({len(batch_texts)} articles)")
            start_time = time()
            
            try:
                # Generate summaries
                summaries = generate_summaries_batch(
                    model, tokenizer, batch_texts, batch_prompts, max_summary_length
                )
                
                # Prepare data for saving
                summaries_data = []
                for j, summary in enumerate(summaries):
                    summaries_data.append({
                        "article_id": batch_ids[j],
                        "language": lang,
                        "summary": summary,
                        "title": batch_titles[j]
                    })
                
                # Save summaries
                save_summaries(summaries_dir, summaries_data)
                
                # Update tracking
                for article_id in batch_ids:
                    completed_articles.add(article_id)
                
                # Log performance
                elapsed = time() - start_time
                articles_per_sec = len(batch_texts) / elapsed if elapsed > 0 else 0
                print(f"Generated {len(batch_texts)} summaries in {elapsed:.2f}s ({articles_per_sec:.2f} articles/sec)")
                
                # Update tracking file periodically
                if len(completed_articles) % save_interval == 0:
                    update_tracking_file(lang_summaries_dir, completed_articles)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Try to save what we have so far
                update_tracking_file(lang_summaries_dir, completed_articles)
                
                # Try clearing CUDA cache if there's a memory error
                if "CUDA out of memory" in str(e) and torch.cuda.is_available():
                    print("CUDA out of memory - trying to clear cache")
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Final update to tracking file
        update_tracking_file(lang_summaries_dir, completed_articles)
        print(f"Completed {len(completed_articles)} summaries for {lang}")
    
    print("\nSummary generation complete!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate summaries for multilingual Wikipedia articles")
    
    parser.add_argument("--raw-data-dir", required=True,
                        help="Directory containing raw extracted Wikipedia data")
    parser.add_argument("--summaries-dir", required=True,
                        help="Output directory for generated summaries")
    parser.add_argument("--languages", nargs="+", default=["en", "fr", "de", "ja", "ru"],
                        help="Languages to process (default: en fr de ja ru)")
    parser.add_argument("--model-name", required=True,
                        help="LLM model to use for summary generation (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--max-input-length", type=int, default=4096,
                        help="Maximum input length for the LLM")
    parser.add_argument("--max-summary-length", type=int, default=512,
                        help="Maximum summary length to generate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing articles")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Maximum number of articles to process (for testing)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resuming from previous run")
    parser.add_argument("--precision", choices=["4bit", "8bit", "16bit", "32bit"], default="4bit",
                        help="Model precision (default: 4bit)")
    parser.add_argument("--gpu-memory", type=float, default=0.7,
                        help="Fraction of GPU memory to use (0-1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="How often to update tracking file")
    
    args = parser.parse_args()
    
    generate_and_save_summaries(
        args.raw_data_dir,
        args.summaries_dir,
        args.languages,
        args.model_name,
        args.max_input_length,
        args.max_summary_length,
        args.batch_size,
        args.max_articles,
        not args.no_resume,
        args.precision,
        args.gpu_memory,
        args.save_interval
    )


if __name__ == "__main__":
    main()