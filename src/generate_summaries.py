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
        prompt = f"""Extract the essential information from this article and create a clear, focused summary in English.
        As an expert content analyst, identify the main arguments, key points, and significant conclusions.
        Present the information in a straightforward, objective manner that respects the original perspective.
        Output only the summary.

        Article:
        {text}

        Summary:"""

    elif language == "fr":
        prompt = f"""Extrayez les informations essentielles de cet article et créez un résumé clair et ciblé en français.
        En tant qu'analyste de contenu expert, identifiez les arguments principaux, les points clés et les conclusions importantes.
        Présentez l'information de manière directe et objective qui respecte la perspective originale.
        Ne produisez que le résumé.

        Article:
        {text}

        Résumé:"""

    elif language == "de":
        prompt = f"""Extrahieren Sie die wesentlichen Informationen aus diesem Artikel und erstellen Sie eine klare, fokussierte Zusammenfassung auf Deutsch.
        Als Experte für Inhaltsanalyse identifizieren Sie die Hauptargumente, Schlüsselpunkte und wichtigen Schlussfolgerungen.
        Präsentieren Sie die Informationen in einer direkten, objektiven Weise, die die ursprüngliche Perspektive respektiert.
        Geben Sie nur die Zusammenfassung aus.

        Artikel:
        {text}

        Zusammenfassung:"""
        
    elif language == "ja":
        prompt = f"""この記事から重要な情報を抽出し、日本語で明確で焦点を絞った要約を作成してください。
        コンテンツ分析の専門家として、主な議論、重要なポイント、重要な結論を特定してください。
        元の視点を尊重する、直接的で客観的な方法で情報を提示してください。
        要約のみを出力してください。

        記事:
        {text}

        要約:"""

    elif language == "ru":
        prompt = f"""Извлеките существенную информацию из этой статьи и создайте четкое, целенаправленное резюме на русском языке.
        Как эксперт по анализу контента, определите основные аргументы, ключевые моменты и важные выводы.
        Представьте информацию прямым, объективным способом, который уважает исходную перспективу.
        Выводите только резюме.

        Статья:
        {text}

        Резюме:"""

    else:
        prompt = f"""Extract the essential information from this article and create a clear, focused summary in {language}.
        As an expert content analyst, identify the main arguments, key points, and significant conclusions.
        Present the information in a straightforward, objective manner that respects the original perspective.
        Output only the summary.

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