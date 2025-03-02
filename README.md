# Multilingual Summarization Fine-Tuning Project

This project implements a modular pipeline for fine-tuning Small Language Models (SLMs) on multilingual summarization tasks using parallel Wikipedia articles.

## Project Structure

The codebase is organized into modular components:

### Data Processing
- `dataset.py` - Extract Wikipedia articles (in all 5 languages parallel)
- `generate_summaries.py` - Generate summaries using a large language model
- `convert_to_hf_dataset.py` - Create HuggingFace datasets from raw data and summaries

### Model Training and Evaluation
- `finetune_model.py` - Fine-tune a model on the summarization task
- `evaluate_model.py` - Evaluate model performance
- `compare_baselines.py` - Compare multiple models

### Utility Modules
- `model_utils.py` - Model loading/saving utilities
- `data_utils.py` - Data processing utilities
- `metrics_utils.py` - Metrics computation and visualization

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage Workflow

### 1. Data Extraction and Preparation

First, extract parallel Wikipedia articles:

```bash
python dataset.py --languages fr de ja ru --num-documents 5000 --output-dir Data/directory/raw
```

Next, generate summaries for each language:

```bash
python generate_summaries.py \
  --raw-data-dir Data/directory/raw \
  --summaries-dir Data/directory/summaries \
  --languages fr de ja ru \
  --model-name Qwen/Qwen2.5-7B-Instruct
```

Finally, convert the raw data and summaries to a HuggingFace dataset:

```bash
python convert_to_hf_dataset.py \
  --raw-data-dir Data/directory/raw \
  --summaries-dir Data/directory/summaries \
  --hf-dataset-dir Data/directory/hf_dataset \
  --languages fr de ja ru
```

### 2. Model Fine-tuning

Fine-tune a small language model on a specific language:

```bash
python finetune_model.py \
  --model-name google/mt5-base \
  --dataset-path Data/directory/hf_dataset \
  --language fr \
  --output-dir models \
  --batch-size 4 \
  --num-epochs 3
```

Options:
- Use `--no-lora` to disable LoRA (use full fine-tuning)
- Use `--lora-r` and `--lora-alpha` to configure LoRA parameters
- Use `--learning-rate` to adjust the learning rate

### 3. Model Evaluation

Evaluate the fine-tuned model:

```bash
python evaluate_model.py \
  --model-path models/fr_mt5-base_20250302_123456/final_model \
  --dataset-path Data/directory/hf_dataset \
  --language fr
```

For LoRA models, use:
```bash
python evaluate_model.py \
  --model-path models/fr_mt5-base_lora_20250302_123456/final_model \
  --dataset-path Data/directory/hf_dataset \
  --language fr \
  --lora \
  --base-model google/mt5-base
```

### 4. Model Comparison

Compare your fine-tuned model against baselines:

```bash
python compare_baselines.py \
  --config model_config.json \
  --dataset-path Data/directory/hf_dataset \
  --language fr \
  --output-dir model_comparison
```

Where `model_config.json` defines the models to compare:

```json
{
  "models": [
    {
      "name": "google/mt5-base",
      "type": "huggingface",
      "display_name": "mT5-Base (Baseline)"
    },
    {
      "name": "./models/fr_mt5-base_20250302_123456/final_model",
      "type": "finetuned",
      "display_name": "Fine-tuned mT5-Base"
    }
  ]
}
```

## Running on Google Colab

For Colab compatibility:
1. Keep batch sizes small (2-4) to fit in GPU memory
2. Use LoRA fine-tuning (enabled by default)
3. For large models, reduce `max_input_length` and `max_target_length`

Example Colab setup:
```python
# Install requirements
!pip install -r requirements.txt

!git clone https://github.com/shashuat/multilingual-summarization.git
%cd multilingual-summarization

# Run the fine-tuning
!python finetune_model.py \
  --model-name google/mt5-base \
  --dataset-path ./Data/directory/hf_dataset \
  --language fr \
  --batch-size 2 \
  --num-epochs 3
```

## Recommended Small Language Models

Models that work well on Google Colab (16GB GPU):
- `google/mt5-base` (580M parameters, multilingual)


## Evaluation Metrics

Models are evaluated using:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore (precision, recall, F1)
- Length statistics and correlation
- Visualizations of performance patterns

## Project Structure

```
├── data/
│   ├── raw/               # Raw Wikipedia articles
│   ├── summaries/         # Generated summaries
│   └── hf_dataset/        # Processed HuggingFace datasets
├── models/                # Fine-tuned models
├── model_comparison/      # Model comparison results
├── dataset.py             # Wikipedia data extraction script
├── generate_summaries.py  # Summary generation script
├── convert_to_hf_dataset.py # Dataset conversion script
├── model_utils.py         # Model utilities
├── data_utils.py          # Data processing utilities
├── metrics_utils.py       # Metrics computation utilities
├── finetune_model.py      # Fine-tuning script
├── evaluate_model.py      # Evaluation script
├── compare_baselines.py   # Model comparison script
└── requirements.txt       # Project dependencies
```