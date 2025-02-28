#!/usr/bin/env python3
"""
dataset.py - Extract parallel Wikipedia articles in multiple languages

This script extracts Wikipedia articles that have versions in all specified languages,
saving them with a consistent naming convention to show correspondence.
"""

import os
import json
import time
import argparse
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import mwclient
from tqdm import tqdm
import pandas as pd


@dataclass
class WikiConfig:
    """Configuration for Wikipedia parallel data extraction"""
    languages: List[str] = field(default_factory=lambda: ["en", "fr", "de", "ja", "ru"])
    num_documents: int = 10000
    output_dir: str = "data/raw"
    # Categories to extract from (will try these in order until we have enough documents)
    categories: List[str] = field(default_factory=lambda: [
        "Featured_articles", 
        "Good_articles",
        "Science",
        "Philosophy",
        "Culture",
        "Politics",
        "History",
        "Geography",
        "Religion",
        "Mathematics",
        "Biology",
        "Chemistry",
        "Physics",
        "Astronomy",
        "Psychology",
        "Technology",
    ])
    # How many articles to get from each category
    articles_per_category: int = 30000
    # Min content length to consider (to filter out stubs)
    min_content_length: int = 1000
    # Delay between requests to avoid hitting rate limits (in seconds)
    request_delay: float = 0.1
    # Save progress every N documents
    save_checkpoint_every: int = 100
    # Resume from checkpoint if available
    resume_from_checkpoint: bool = True


class WikiParallelExtractor:
    """Extract parallel Wikipedia articles in multiple languages"""
    
    def __init__(self, config: WikiConfig):
        """Initialize with the given configuration"""
        self.config = config
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Dictionary to track processed pages to avoid duplicates
        self.processed_pages: Set[str] = set()
        
        # Load checkpoint if available and resume_from_checkpoint is True
        self.checkpoint_path = os.path.join(self.config.output_dir, "checkpoint.json")
        self.article_index_path = os.path.join(self.config.output_dir, "article_index.json")
        
        if self.config.resume_from_checkpoint and os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                self.processed_pages = set(checkpoint["processed_pages"])
                print(f"Resuming from checkpoint with {len(self.processed_pages)} processed pages")
        
        # Initialize article index
        self.article_index = []
        if os.path.exists(self.article_index_path):
            with open(self.article_index_path, "r", encoding="utf-8") as f:
                self.article_index = json.load(f)
        
        # Initialize mwclient sites
        print(f"Initializing Wikipedia clients for languages: {', '.join(self.config.languages)}")
        self.sites = {}
        for lang in self.config.languages:
            self.sites[lang] = mwclient.Site(f'{lang}.wikipedia.org')
    
    def extract_data(self) -> List[Dict]:
        """Extract parallel documents from Wikipedia"""
        parallel_docs = []
        
        # Process each category
        for category in self.config.categories:
            print(f"\nExtracting from category: {category}")
            
            # Get base pages from English Wikipedia category
            try:
                base_pages = list(self.sites["en"].Categories[category])[:self.config.articles_per_category]
            except Exception as e:
                print(f"Error getting category {category}: {e}")
                continue
            
            # Process each page
            for page in tqdm(base_pages, desc=f"Processing {category}"):
                # Skip if we've already processed this page
                if page.name in self.processed_pages:
                    continue
                
                # Mark as processed so we don't try it again
                self.processed_pages.add(page.name)
                
                # Skip non-article pages
                if page.namespace != 0:
                    continue
                
                # Get parallel content
                try:
                    parallel_set = self._get_parallel_content(page)
                    if parallel_set:
                        article_id = self._generate_article_id(page.name)
                        parallel_set["id"] = article_id
                        parallel_docs.append(parallel_set)
                        
                        # Save individual language files
                        self._save_article(parallel_set)
                        
                        # Update article index
                        self._update_article_index(parallel_set)
                        
                        # Save checkpoint periodically
                        if len(parallel_docs) % self.config.save_checkpoint_every == 0:
                            self._save_checkpoint()
                            print(f"Saved checkpoint with {len(parallel_docs)} documents")
                except Exception as e:
                    print(f"Error processing {page.name}: {e}")
                
                # Respect rate limits
                time.sleep(self.config.request_delay)
                
                # Break if we have enough documents
                if len(parallel_docs) >= self.config.num_documents:
                    break
            
            # Break if we have enough documents
            if len(parallel_docs) >= self.config.num_documents:
                break
        
        # Final checkpoint
        self._save_checkpoint()
        
        print(f"\nExtracted {len(parallel_docs)} parallel documents")
        return parallel_docs
    
    def _get_parallel_content(self, base_page) -> Optional[Dict]:
        """Get content of the same page in all languages"""
        # Get langlinks for the page
        try:
            langlinks_raw = list(base_page.langlinks())
            
            # Process langlinks based on their actual structure
            # In current mwclient versions, they're tuples like (lang_code, title)
            langlink_map = {}
            for link in langlinks_raw:
                if isinstance(link, tuple) and len(link) >= 2:
                    lang_code, title = link
                    langlink_map[lang_code] = title
                elif hasattr(link, 'lang') and hasattr(link, 'title'):
                    # Older mwclient versions might return objects
                    langlink_map[link.lang] = link.title
                else:
                    # Skip unrecognized format
                    continue
                    
        except Exception as e:
            print(f"Error getting langlinks for {base_page.name}: {e}")
            return None
        
        # Extract base page content
        base_content = self._extract_content(base_page)
        
        # Skip if content is too short (probably a stub)
        if len(base_content["text"]) < self.config.min_content_length:
            return None
        
        parallel_set = {
            "en": base_content
        }
        
        # Get content for each target language
        for lang in self.config.languages:
            if lang == "en":
                continue  # Already got English content
            
            if lang not in langlink_map:
                return None  # Skip if any language is missing
            
            # Get page in target language
            title = langlink_map[lang]
            try:
                foreign_page = self.sites[lang].pages[title]
                foreign_content = self._extract_content(foreign_page)
                
                # Skip if content is too short
                if len(foreign_content["text"]) < self.config.min_content_length:
                    return None
                
                parallel_set[lang] = foreign_content
            except Exception as e:
                print(f"Error getting {lang} page {title}: {e}")
                return None
        
        # Return the parallel set if we have content for all languages
        if all(lang in parallel_set for lang in self.config.languages):
            return parallel_set
        return None
    
    def _extract_content(self, page) -> Dict:
        """Extract content from a page"""
        # Construct the URL manually since page.fullurl doesn't exist
        url = f"https://{page.site.host}/wiki/{page.name.replace(' ', '_')}"
        return {
            "title": page.name,
            "text": page.text(),
            "url": url,
            "pageid": page.pageid
        }
    
    def _generate_article_id(self, title: str) -> str:
        """Generate a unique ID for the article based on its title"""
        hash_object = hashlib.md5(title.encode())
        return hash_object.hexdigest()[:10]
    
    def _save_article(self, parallel_set: Dict) -> None:
        """Save article content to files with a consistent naming convention"""
        article_id = parallel_set["id"]
        
        for lang, content in parallel_set.items():
            if lang == "id":
                continue  # Skip the ID field
            
            # Create language directory if it doesn't exist
            lang_dir = os.path.join(self.config.output_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            
            # Save content to file
            filename = f"{article_id}.json"
            filepath = os.path.join(lang_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
    
    def _update_article_index(self, parallel_set: Dict) -> None:
        """Update the article index with metadata about this article"""
        article_id = parallel_set["id"]
        
        # Create index entry
        index_entry = {
            "id": article_id,
            "titles": {lang: content["title"] for lang, content in parallel_set.items() if lang != "id"},
            "urls": {lang: content["url"] for lang, content in parallel_set.items() if lang != "id"},
        }
        
        # Add to index
        self.article_index.append(index_entry)
        
        # Save index
        with open(self.article_index_path, "w", encoding="utf-8") as f:
            json.dump(self.article_index, f, ensure_ascii=False, indent=2)
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint of processed pages"""
        checkpoint = {
            "processed_pages": list(self.processed_pages)
        }
        
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False)


def main():
    """Main function to run the extraction process"""
    parser = argparse.ArgumentParser(description="Extract parallel Wikipedia articles")
    
    parser.add_argument("--languages", nargs="+", default=["en", "fr", "de", "ja", "ru"],
                        help="Languages to extract (default: en fr de ja ru)")
    parser.add_argument("--num-documents", type=int, default=10000,
                        help="Number of documents to extract (default: 10000)")
    parser.add_argument("--output-dir", default="data/raw",
                        help="Output directory (default: data/raw)")
    parser.add_argument("--min-content-length", type=int, default=1000,
                        help="Minimum content length to consider (default: 1000)")
    
    args = parser.parse_args()
    
    config = WikiConfig(
        languages=args.languages,
        num_documents=args.num_documents,
        output_dir=args.output_dir,
        min_content_length=args.min_content_length
    )
    
    extractor = WikiParallelExtractor(config)
    parallel_docs = extractor.extract_data()
    
    # Create metadata file with overall statistics
    metadata = {
        "total_documents": len(parallel_docs),
        "languages": config.languages,
        "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_documents": config.num_documents,
            "min_content_length": config.min_content_length,
            "categories": config.categories
        }
    }
    
    metadata_path = os.path.join(config.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\nExtraction complete. Data saved to {config.output_dir}")
    print(f"Total documents: {len(parallel_docs)}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()