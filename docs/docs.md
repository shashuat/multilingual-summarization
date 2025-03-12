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
python -m convert_to_hf_dataset2 \
  --raw-data-dir data/raw \
  --summaries-dir data/summaries \
  --hf-dataset-dir /Users/shash/hub/mult_test \
  --languages fr de ja ru

python -m src.generate_summaries \
  --raw-data-dir /Data/shash/mul/raw \
  --summaries-dir /Data/shash/mul/summaries_test \
  --languages fr de en \
  --model-name mistralai/Mistral-Small-24B-Instruct-2501 \
  --max-articles 6000

python -m src.generate_summaries \
  --raw-data-dir /Data/shash/mul/raw \
  --summaries-dir /Data/shash/mul/summaries_test4 \
  --languages fr \
  --model-name microsoft/Phi-4-mini-instruct \
  --max-articles 6000
  
mistralai/Mistral-Small-24B-Instruct-2501
CohereForAI/aya-23-35B
## finetune

### japanese
python -m src.finetune_phi \
  --model_name microsoft/Phi-4-mini-instruct \
  --dataset_path /Data/shash/mul/hf_dataset2 \
  --language ja \
  --output_dir /Data/shash/mul/finetuned_models_ja4_phi_32_ep5 \
  --wandb_project mulsum-phi

output: INFO:__main__:Loading model and tokenizer: microsoft/Phi-4-mini-instruct
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.29it/s]
INFO:__main__:Setting up LoRA with r=32, alpha=64
INFO:__main__:trainable params: 11,534,336 || all params: 3,847,556,096 || trainable%: 0.2998
INFO:__main__:Loading dataset from /Data/shash/mul/hf_dataset2/ja
INFO:__main__:Loaded 3999 training examples and 501 validation examples
INFO:__main__:Preprocessing datasets
INFO:__main__:Adding ROUGE evaluation callback with 10 samples
INFO:__main__:Starting training

### french
python -m src.finetune_phi \
  --model_name microsoft/Phi-4-mini-instruct \
  --dataset_path /Data/shash/mul/hf_dataset2 \
  --language fr \
  --output_dir /Data/shash/mul/finetuned_models256_fr4_phi_32_ep5 \
  --wandb_project mulsum-phi

output: INFO:__main__:Loading model and tokenizer: microsoft/Phi-4-mini-instruct
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.44it/s]
INFO:__main__:Setting up LoRA with r=32, alpha=64
INFO:__main__:trainable params: 11,534,336 || all params: 3,847,556,096 || trainable%: 0.2998
INFO:__main__:Loading dataset from /Data/shash/mul/hf_dataset2/fr
INFO:__main__:Loaded 3999 training examples and 501 validation examples
INFO:__main__:Preprocessing datasets
INFO:__main__:Adding ROUGE evaluation callback with 10 samples
INFO:__main__:Starting training 

12678MiB

### german
python -m src.finetune_phi \
  --model_name microsoft/Phi-4-mini-instruct \
  --dataset_path /Data/shash/mul/hf_dataset2 \
  --language de \
  --output_dir /Data/shash/mul/finetuned_models256_de4_phi_32_ep5 \
  --wandb_project mulsum-phi \
  --max_length 4096 \
  --lora_r 32 \
  --lora_alpha 64
output: INFO:__main__:Loading model and tokenizer: microsoft/Phi-4-mini-instruct
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|██████████████████████████████████████| 2/2 [00:01<00:00,  1.27it/s]
INFO:__main__:Setting up LoRA with r=32, alpha=64
INFO:__main__:trainable params: 11,534,336 || all params: 3,847,556,096 || trainable%: 0.2998
INFO:__main__:Loading dataset from /Data/shash/mul/hf_dataset2/de
INFO:__main__:Loaded 3999 training examples and 501 validation examples
INFO:__main__:Preprocessing datasets
Preprocessing dataset: 100%|██████████████████████████████| 501/501 [00:02<00:00, 238.52 examples/s]
INFO:__main__:Adding ROUGE evaluation callback with 10 samples
INFO:__main__:Starting training

10gb for 1024 length, 12 gb for 4096 length lora_2 = 8

12890MiB for 4096 and lora_r = 32

500/1245 [2:42:08<2:53:03, 13.94s/it}
Calculating ROUGE metrics on validation set...███████| 501/501 [02:49<00:00,  3.44it/s]
{'eval_loss': 0.6197781562805176, 'eval_runtime': 170.1085, 'eval_samples_per_second': 2.945, 'eval_steps_per_second': 2.945, 'epoch': 2.0}
## evaluate

python -m src.eval_llama \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --dataset_path /Data/shash/mul/hf_dataset2 \
  --language fr \
  --model_path /Data/shash/mul/finetuned_models

## compare

python -m src.compare_llama \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --finetuned_model /Data/shash/mul/finetuned_models \
    --dataset_path /Data/shash/mul/hf_dataset2 \
    --language fr \
    --num_samples 5 \
    --output_file comparison_results.json

python -m src.compare_phi \
  --base_model "microsoft/Phi-4-mini-instruct" \
  --finetuned_model "/Data/shash/mul/finetuned_models_fr4_phi_32_ep5/checkpoint-1245" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "en" \
  --num_samples 20 \
  --subset "test" \
  --output_file "comparison_results_fr4_phi_32_ep5-1245-fr-en.json"

python -m src.compare_phi \
  --base_model "microsoft/Phi-4-mini-instruct" \
  --finetuned_model "/Data/shash/mul/finetuned_models_de4_phi_32_ep5/checkpoint-1245" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "en" \
  --num_samples 20 \
  --subset "test" \
  --output_file "comparison_results_de4_phi_32_ep5-1245-de-en.json"
