# Why __init__.py is required
We need to set the HuggingFace environment variable to correspond to the larger shared drive, otherwise when we download LLM models we will run out of disk space and get 'disk quota exceeded' error.

```
os.environ["HF_HOME"] = "/Data/shash/.cache"
os.environ["HF_HUB_CACHE"] = "/Data/shash/.cache/hub"
```

Then I just load my hf and wandb tokens to login from the .env file.
HF_TOKEN=hf_something
WANDB_TOKEN=e58_something

# To run the generate_summaries.py file for a model 
(example Qwen2.5-32B-Instruct)

```
python -m src.generate_summaries \
  --raw-data-dir /Data/shash/mul/raw \
  --summaries-dir /Data/shash/mul/summaries_test \
  --languages de \
  --model-name Qwen/Qwen2.5-32B-Instruct
```
python -m convert_to_hf_dataset \
  --raw-data-dir data/raw \
  --summaries-dir data/summaries \
  --hf-dataset-dir /Users/shash/hub/mult_test \
  --languages fr de ja ru

python -m src.generate_summaries_batch \
  --raw-data-dir /Data/shash/mul/raw \
  --summaries-dir /Data/shash/mul/summaries_test \
  --languages fr de \
  --model-name Qwen/Qwen2.5-32B-Instruct \
  --batch-size 8 \
  --precision 4bit

python -m src.generate_summaries \
  --raw-data-dir /Data/shash/mul/raw \
  --summaries-dir /Data/shash/mul/summaries_test \
  --languages fr \
  --model-name mistralai/Mistral-Small-24B-Instruct-2501

## finetune

python -m src.finetune_llama \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --dataset_path /Data/shash/mul/hf_dataset \
  --language fr \
  --output_dir /Data/shash/mul/finetuned_models

