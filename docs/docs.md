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
(example Qwen2.5-7B-Instruct)

```
python -m src.generate_summaries \
  --raw-data-dir /Data/shash/mul/raw \
  --summaries-dir /Data/shash/mul/summaries_test \
  --languages fr de ja ru \
  --model-name Qwen/Qwen2.5-14B-Instruct
```
python convert_to_hf_dataset.py \
  --raw-data-dir data/raw \
  --summaries-dir data/summaries \
  --hf-dataset-dir /Users/shash/hub/mult_test \
  --languages fr de ja ru

python -m src.convert_to_hf_dataset \
  --raw-data-dir /Data/shash/mul/raw \
  --summaries-dir /Data/shash/mul/summaries_test \
  --hf-dataset-dir /Data/shash/mul/mult_test \
  --languages fr

Mistral-Small-24B-Instruct-2501