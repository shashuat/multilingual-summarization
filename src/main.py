# main.py
from config import DataConfig, ModelConfig, TrainingConfig
from dataset import WikiParallelExtractor, DatasetCreator
from models import QuantizedQwenSummarizer, MT5Summarizer

if __name__ == "__main__":
    # Load configurations
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Create dataset
    extractor = WikiParallelExtractor(data_config)
    parallel_docs = extractor.extract_parallel_documents()
    
    # Initialize models
    summary_model = QuantizedQwenSummarizer(model_config)
    mt5_model = MT5Summarizer(model_config)
    
    # Create datasets with synthetic summaries
    dataset_creator = DatasetCreator(data_config, summary_model)
    datasets = dataset_creator.create_datasets(parallel_docs)
    
    # Train models for each language
    # for lang in data_config.languages:
    #     trainer = SummaryTrainer(mt5_model, train_config, model_config)
    #     trainer.train(
    #         datasets[lang]['train'],
    #         datasets[lang]['test']
    #     )
        
    #     # Evaluate
    #     metrics = trainer.evaluate(datasets[lang]['test'])
    #     print(f"Metrics for {lang}:", metrics)