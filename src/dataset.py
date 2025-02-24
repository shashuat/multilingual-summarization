# dataset.py
import mwclient
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict, List
from config import DataConfig
import os

class WikiParallelExtractor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.sites = {
            lang: mwclient.Site(f'{lang}.wikipedia.org')
            for lang in config.languages
        }
        
    def extract_parallel_documents(self) -> List[Dict]:
        base_pages = self.sites['en'].categories[self.config.wiki_category]
        parallel_docs = []
        
        for page in tqdm(base_pages, total=self.config.num_documents):
            if len(parallel_docs) >= self.config.num_documents:
                break
                
            parallel_set = self._get_parallel_content(page)
            if parallel_set:
                parallel_docs.append(parallel_set)
                
        return parallel_docs
    
    def _get_parallel_content(self, base_page) -> Optional[Dict]:
        langlinks = base_page.langlinks()
        parallel_set = {
            'en': self._extract_content(base_page)
        }
        
        for link in langlinks:
            if link.lang in self.config.languages:
                foreign_page = self.sites[link.lang].pages[link.title]
                parallel_set[link.lang] = self._extract_content(foreign_page)
                
        if all(lang in parallel_set for lang in self.config.languages):
            return parallel_set
        return None
        
    def _extract_content(self, page) -> Dict:
        return {
            'title': page.page_title,
            'text': page.text(),
            'url': page.full_url()
        }

class DatasetCreator:
    def __init__(self, config: DataConfig, summarizer: BaseSummarizer):
        self.config = config
        self.summarizer = summarizer
        
    def create_datasets(self, parallel_docs: List[Dict]) -> Dict[str, DatasetDict]:
        datasets = {}
        
        for lang in self.config.languages:
            # Prepare data for current language
            data = []
            for doc_set in tqdm(parallel_docs):
                summary = self.summarizer.generate_summary(doc_set['en']['text'])
                data.append({
                    'text': doc_set[lang]['text'],
                    'summary': summary,
                    'title': doc_set[lang]['title'],
                    'url': doc_set[lang]['url']
                })
                
            # Create dataset
            dataset = Dataset.from_pandas(pd.DataFrame(data))
            split_dataset = dataset.train_test_split(
                train_size=self.config.train_ratio,
                shuffle=True,
                seed=42
            )
            
            datasets[lang] = DatasetDict({
                'train': split_dataset['train'],
                'test': split_dataset['test']
            })
            
            # Save dataset
            output_dir = os.path.join(self.config.data_dir, f"wiki_summaries_{lang}")
            datasets[lang].save_to_disk(output_dir)
            
        return datasets
