# multilingual-summarization

## Installation
```
git lfs install
git lfs track "data/*"
git lfs track "outputs/*"
```

### 1. Configuration Management (`config.py`)

#### DataConfig
- `languages`: List[str] - Target languages for parallel corpus creation
- `wiki_category`: str - Source Wikipedia category for data extraction
- `num_documents`: int - Number of documents to extract (default: 5000)
- `train_ratio`: float - Train/test split ratio
- `max_input_length`: int - Maximum token length for input text
- `max_summary_length`: int - Maximum token length for generated summaries

#### ModelConfig
- Base Model Parameters:
  - `base_model_name`: str - Name of the base mT5 model
  - `summary_model_name`: str - Name of the synthetic summary generation model
  - `use_quantization`: bool - Toggle for model quantization
  - `quantization_bits`: int - Bits for quantization (default: 4)

- LoRA Hyperparameters:
  - `lora_r`: int - LoRA rank (default: 16)
  - `lora_alpha`: int - LoRA alpha scaling factor (default: 32)
  - `lora_dropout`: float - Dropout probability
  - `target_modules`: List[str] - Target modules for LoRA adaptation

- Training Hyperparameters:
  - `learning_rate`: float - Initial learning rate
  - `num_epochs`: int - Number of training epochs
  - `train_batch_size`: int - Training batch size
  - `gradient_accumulation_steps`: int - Steps for gradient accumulation
  - Generation parameters (temperature, top_p, etc.)

### 2. Model Architecture (`models.py`)

#### BaseSummarizer
Abstract base class defining the summarizer interface:
```python
def generate_summary(self, text: str) -> str:
    """Generate summary for input text"""
    raise NotImplementedError
```

#### MT5Summarizer
Implements the base summarizer using mT5:
- Initialization:
  ```python
  def __init__(self, config: ModelConfig):
      self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
      self.model = AutoModelForSeq2SeqGeneration.from_pretrained(...)
  ```
- LoRA Integration:
  - Uses PEFT library for parameter-efficient fine-tuning
  - Targets query and value matrices in transformer blocks
  - Reduces trainable parameters by ~99%

#### QuantizedQwenSummarizer
Implements quantized Qwen model for synthetic summary generation:
- Uses AutoGPTQ for 4-bit quantization
- Implements efficient inference with quantized weights
- Custom prompt template for summary generation

### 3. Dataset Processing (`dataset.py`)

#### WikiParallelExtractor
Handles parallel corpus creation:
```python
def extract_parallel_documents(self) -> List[Dict]:
    """Extract aligned documents across languages"""
```
- Uses mwclient for efficient Wikipedia API access
- Implements parallel processing for extraction
- Maintains language alignment through langlinks

#### DatasetCreator
Manages dataset preparation:
- Synthetic summary generation using quantized model
- Dataset splitting and preprocessing
- HuggingFace Dataset integration

### 4. Training Pipeline (`train.py`)

#### SummaryTrainer
Implements training loop and evaluation:
```python
def train(self, train_dataset, eval_dataset):
    """Train model with specified datasets"""
```

Key Features:
- Gradient accumulation for memory efficiency
- Mixed precision training (FP16)
- Progressive learning rate scheduling
- Evaluation metrics computation (ROUGE, BERTScore)

### 5. Generation Interface (`generate.py`)

Provides inference interface:
```python
def generate_summaries(
    texts: List[str],
    model_config: ModelConfig,
    checkpoint_path: str = None
) -> List[str]
```

## Technical Implementation Details

### Memory Optimization
1. Gradient Accumulation:
   ```python
   gradient_accumulation_steps = batch_size // micro_batch_size
   ```

2. Mixed Precision Training:
   - Uses torch.amp for automatic mixed precision
   - FP16 for computation, FP32 for accumulation

### Model Quantization
1. AutoGPTQ Implementation:
   ```python
   model = AutoGPTQForCausalLM.from_quantized(
       model_name,
       use_safetensors=True,
       device_map="auto"
   )
   ```

2. LoRA Configuration:
   ```python
   lora_config = LoraConfig(
       r=16,              # Low-rank approximation dimension
       lora_alpha=32,     # Scaling factor
       target_modules=["q", "v"]  # Target attention matrices
   )
   ```

### Dataset Processing Pipeline

1. Parallel Document Extraction:
   ```python
   parallel_set = {
       lang: self._extract_content(page)
       for lang, page in self._get_parallel_pages(base_page)
   }
   ```

2. Content Cleaning:
   - Removes Wiki markup
   - Handles special characters
   - Normalizes whitespace

### Evaluation Metrics

1. ROUGE Score Implementation:
   ```python
   rouge = evaluate.load('rouge')
   metrics = rouge.compute(
       predictions=predictions,
       references=references
   )
   ```

2. BERTScore for Semantic Similarity:
   - Cross-lingual evaluation capability
   - Contextual embeddings for better semantic matching

## Performance Considerations

### GPU Memory Usage
- Base mT5 model: ~2GB
- Quantized Qwen: ~4GB
- Total pipeline: ~8GB peak memory

### Training Time Estimates
- Data extraction: ~2-3 hours
- Training (per language): ~4-6 hours on T4 GPU
- Inference: ~100ms per summary

## Future Improvements

1. Model Architecture:
   - Implement model distillation
   - Explore more efficient attention mechanisms

2. Training Optimization:
   - Implementation of 8-bit training
   - Dynamic batch sizing
   - Curriculum learning strategy

3. Dataset Enhancement:
   - Quality filtering based on article metrics
   - Cross-lingual consistency checking
   - Data augmentation techniques

4. LLM as a Judge:
   - Use LLM to judge the quality of the summaries
   - Use the LLM to generate summaries for the data

