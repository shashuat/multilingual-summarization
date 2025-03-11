#!/usr/bin/env python3
"""
generate_summaries.py - Generate summaries for multilingual Wikipedia articles using LLMs

This script loads raw Wikipedia articles, generates summaries using an LLM,
and saves the summaries to disk for later use in dataset creation.
"""

import os
import json
import argparse
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def setup_llm(model_name: str):
    """Set up the LLM for summary generation with appropriate quantization"""
    print(f"Loading {model_name} for summary generation...")
    
    # Configure quantization 
    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    memory_used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"GPU memory used: {memory_used:.2f} GB")
    
    return model, tokenizer

def generate_summary(model, tokenizer, text: str, language: str, 
                     max_input_length: int = 4096, 
                     max_summary_length: int = 512) -> str:
    """Generate a summary for a single article using the LLM"""
    # Truncate input if needed
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
        # English for other languages, shouldn't happen becasue we only use fr de en ja ru, but for completeness
        prompt = f"""Please provide a concise summary of the following article in {language}. 
        The summary should be comprehensive, capturing all key points and main arguments, 
        but avoid unnecessary details. Output only the summary.

        Article:
        {text}

        Summary:"""

    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_summary_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            attention_mask=inputs.attention_mask
        )
    
    # Decode and extract only the summary part (removing the prompt)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = full_output[len(prompt):].strip()
    
    return summary


def generate_and_save_summaries(
    raw_data_dir: str,
    summaries_dir: str,
    languages: List[str],
    model_name: str,
    max_input_length: int = 4096,
    max_summary_length: int = 512,
    batch_size: int = 1,
    max_articles: Optional[int] = None,
    resume: bool = True
) -> None:
    """Generate summaries for articles and save them to disk"""
    # Create output directory for summaries
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Load article index
    print(f"Loading article index from {raw_data_dir}...")
    articles = load_article_index(raw_data_dir)
    
    if max_articles:
        articles = articles[:max_articles]
        
    print(f"Found {len(articles)} articles with parallel content")
    
    # Set up LLM for summary generation
    model, tokenizer = setup_llm(model_name)
    
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
            
            for article in tqdm(batch, desc=f"Generating {lang} summaries (batch {i//batch_size + 1})"):
                article_id = article["id"]
                
                # Skip if already completed
                if article_id in completed_articles:
                    continue
                
                # Load article content
                article_data = load_article_content(raw_data_dir, article_id, lang)
                
                if not article_data or not article_data["text"].strip():
                    print(f"Skipping article {article_id}: Missing {lang} content")
                    continue
                
                # Generate summary
                try:
                    summary = generate_summary(
                        model, tokenizer, article_data["text"], lang,
                        max_input_length, max_summary_length
                    )
                    
                    # Create summary data structure
                    summary_data = {
                        "article_id": article_id,
                        "language": lang,
                        "summary": summary,
                        "title": article_data["title"]
                    }
                    
                    # Save summary to file
                    summary_path = os.path.join(lang_summaries_dir, f"{article_id}.json")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary_data, f, ensure_ascii=False, indent=2)
                    
                    # Mark as completed
                    completed_articles.add(article_id)
                    
                    # Update tracking file periodically (every 10 articles)
                    if len(completed_articles) % 10 == 0:
                        with open(tracking_file, "w", encoding="utf-8") as f:
                            json.dump({
                                "completed_article_ids": list(completed_articles),
                                "total_completed": len(completed_articles)
                            }, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    print(f"Error generating summary for article {article_id} in {lang}: {e}")
        
        # Final update to tracking file
        with open(tracking_file, "w", encoding="utf-8") as f:
            json.dump({
                "completed_article_ids": list(completed_articles),
                "total_completed": len(completed_articles)
            }, f, ensure_ascii=False, indent=2)
        
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
                        help="LLM model to use for summary generation (e.g., Qwen/Qwen2.5-7B-Instruct, mistralai/Mistral-Small-24B-Instruct-2501)")
    parser.add_argument("--max-input-length", type=int, default=4096,
                        help="Maximum input length for the LLM")
    parser.add_argument("--max-summary-length", type=int, default=512,
                        help="Maximum summary length to generate")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for processing articles")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Maximum number of articles to process (for testing)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resuming from previous run")
    
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
        not args.no_resume
    )


if __name__ == "__main__":
    main()