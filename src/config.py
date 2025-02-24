# config.py
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class DataConfig:
    languages: List[str] = ("en", "hi", "fr", "ja")
    wiki_category: str = "Featured_articles"
    num_documents: int = 5000
    train_ratio: float = 0.8
    max_input_length: int = 512
    max_summary_length: int = 150
    data_dir: str = "data"
    
@dataclass
class ModelConfig:
    # Base model configurations
    base_model_name: str = "google/mt5-base"
    summary_model_name: str = "Qwen/Qwen1.5-7B-Chat"
    use_quantization: bool = True
    quantization_bits: int = 4
    
    # LoRA configurations
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ("q", "v")
    
    # Training configurations
    learning_rate: float = 2e-4
    num_epochs: int = 3
    train_batch_size: int = 4
    eval_batch_size: int = 4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Generation configurations
    num_beams: int = 4
    length_penalty: float = 2.0
    temperature: float = 0.7
    top_p: float = 0.9
    
@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints"
    logging_dir: str = "logs"
    logging_steps: int = 100
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    
