# generate.py
from typing import List
import torch
from config import ModelConfig
from models import MT5Summarizer

def generate_summaries(
    texts: List[str],
    model_config: ModelConfig,
    checkpoint_path: str = None
) -> List[str]:
    """
    Generate summaries for a list of input texts
    """
    # Initialize model
    model = MT5Summarizer(model_config)
    
    # Load fine-tuned weights if provided
    if checkpoint_path:
        model.model.load_state_dict(
            torch.load(checkpoint_path, map_location=model.device)
        )
    
    # Generate summaries
    summaries = []
    for text in texts:
        summary = model.generate_summary(text)
        summaries.append(summary)
        
    return summaries
