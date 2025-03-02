#!/usr/bin/env python3
"""
convert_to_hf_dataset.py - Convert raw Wikipedia parallel data to HuggingFace dataset

This script converts the raw Wikipedia articles and pre-generated summaries into a properly
formatted HuggingFace dataset structured for summarization tasks.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict

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


def load_summary(summaries_dir: str, article_id: str, language: str) -> Optional[str]:
    """Load a generated summary for an article"""
    summary_path = os.path.join(summaries_dir, language, f"{article_id}.json")
    
    if not os.path.exists(summary_path):
        return None
    
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            return summary_data.get("summary", "")
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error loading summary for {article_id} in {language}")
        return None

def create_hf_dataset(
    raw_data_dir: str, 
    summaries_dir: str,
    hf_dataset_dir: str, 
    languages: List[str],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    max_articles: Optional[int] = None,
    seed: int = 42
) -> None:
    """Convert raw data and summaries to HuggingFace dataset format"""
    print(f"Loading article index from {raw_data_dir}...")
    articles = load_article_index(raw_data_dir)
    
    if max_articles:
        articles = articles[:max_articles]
        
    print(f"Found {len(articles)} articles with parallel content")
    
    # Process each language separately
    dataset_dict_by_lang = {}
    
    for lang in languages:
        print(f"\nProcessing {lang} articles...")
        dataset_records = []
        
        for article in tqdm(articles, desc=f"Loading {lang} articles"):
            article_id = article["id"]
            
            # Load article content
            article_data = load_article_content(raw_data_dir, article_id, lang)
            
            if not article_data or not article_data["text"].strip():
                print(f"Skipping article {article_id}: Missing {lang} content")
                continue
            
            # Load summary
            summary = load_summary(summaries_dir, article_id, lang)
            
            if not summary:
                print(f"Skipping article {article_id}: Missing {lang} summary")
                continue
            
            # Add summary to article data
            article_data["summary"] = summary
            
            # Add to dataset records
            dataset_records.append(article_data)
        
        print(f"Successfully processed {len(dataset_records)} articles in {lang}")
        
        # Skip language if no valid articles found
        if len(dataset_records) == 0:
            print(f"No valid articles found for {lang}, skipping...")
            continue
        
        # Convert to DataFrame and then to HuggingFace Dataset
        df = pd.DataFrame(dataset_records)
        lang_dataset = Dataset.from_pandas(df)
        
        # Split into train/validation/test
        # First split off the test set
        test_fraction = test_size / (train_size + val_size + test_size)
        
        train_val_test = lang_dataset.train_test_split(test_size=test_fraction, seed=seed)
        
        # Then split the remaining data into train and validation
        val_fraction = val_size / (train_size + val_size)
        train_val = train_val_test["train"].train_test_split(
            test_size=val_fraction, seed=seed
        )
        
        dataset_dict_by_lang[lang] = DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": train_val_test["test"]
        })
    
    # Create the output directory
    os.makedirs(hf_dataset_dir, exist_ok=True)
    
    # Save each language dataset separately
    for lang, dataset_dict in dataset_dict_by_lang.items():
        lang_dir = os.path.join(hf_dataset_dir, lang)
        print(f"Saving {lang} dataset to {lang_dir}...")
        dataset_dict.save_to_disk(lang_dir)
        
        # Print statistics
        print(f"\n{lang.upper()} dataset statistics:")
        print(f"Total articles: {len(dataset_dict['train']) + len(dataset_dict['validation']) + len(dataset_dict['test'])}")
        print(f"Train set: {len(dataset_dict['train'])} articles")
        print(f"Validation set: {len(dataset_dict['validation'])} articles")
        print(f"Test set: {len(dataset_dict['test'])} articles")
    
    # Create metadata file
    metadata = {
        "description": "Multilingual Wikipedia articles dataset for summarization",
        "languages": languages,
        "article_counts": {
            lang: len(dataset_dict_by_lang[lang]["train"]) + 
                  len(dataset_dict_by_lang[lang]["validation"]) + 
                  len(dataset_dict_by_lang[lang]["test"]) 
            for lang in dataset_dict_by_lang
        },
        "split_ratios": {
            "train": train_size,
            "validation": val_size,
            "test": test_size
        },
        "fields": ["article_id", "language", "title", "text", "summary", "url"]
    }
    
    with open(os.path.join(hf_dataset_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("\nConversion complete!")
    
    # Provide example code for loading the dataset
    print("\nExample code to load the dataset for a specific language:")
    print(f"""
from datasets import load_from_disk

# Load French dataset
fr_dataset = load_from_disk("{hf_dataset_dir}/fr")

# Access train/validation/test splits
train_data = fr_dataset["train"]
val_data = fr_dataset["validation"]
test_data = fr_dataset["test"]

# Example of the first training item
example = train_data[0]
print(f"Title: {{example['title']}}")
print(f"Text length: {{len(example['text'])}}")
print(f"Summary: {{example['summary']}}")
""")

def main():
    """Main function to run the conversion process"""
    parser = argparse.ArgumentParser(description="Convert raw Wikipedia data and summaries to HuggingFace dataset")
    
    parser.add_argument("--raw-data-dir", required=True,
                        help="Directory containing raw extracted Wikipedia data")
    parser.add_argument("--summaries-dir", required=True,
                        help="Directory containing generated summaries")
    parser.add_argument("--hf-dataset-dir", required=True,
                        help="Output directory for the HuggingFace dataset")
    parser.add_argument("--languages", nargs="+", default=["en", "fr", "de", "ja", "ru"],
                        help="Languages to include (default: en fr de ja ru)")
    parser.add_argument("--train-size", type=float, default=0.8,
                        help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Fraction of data for testing (default: 0.1)")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Maximum number of articles to process (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset splitting (default: 42)")
    
    args = parser.parse_args()
    
    # Validate split ratios
    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    create_hf_dataset(
        args.raw_data_dir, 
        args.summaries_dir,
        args.hf_dataset_dir, 
        args.languages,
        args.train_size,
        args.val_size,
        args.test_size,
        args.max_articles,
        args.seed
    )

if __name__ == "__main__":
    main()