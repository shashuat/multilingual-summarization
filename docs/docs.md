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

## finetune Phi

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

/Data/shash/mul/finetuned_models_ja4_phi_32_ep5/checkpoint-1245

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
  --base_model "meta-llama/Llama-3.2-1B-Instruct" \
  --finetuned_model "/Data/shash/mul/finetuned_llama3-2-1b/llama3_1b_fr_ep5/checkpoint-1245" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "en" \
  --num_samples 10 \
  --subset "train" \
  --output_file "comparison_results_256/comparison_results_fr4_llama_32_ep5-1245-fr-fr.json"
  --use_4bit \
  --flash_attention

python -m src.compare_phi \
  --base_model "microsoft/Phi-4-mini-instruct" \
  --finetuned_model "/Data/shash/mul/finetuned_models_fr4_phi_32_ep5/checkpoint-1245" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "en" \
  --num_samples 100 \
  --subset "train" \
  --output_file "comparison_results_256/phi4/comparison_results_fr4_phi_32_ep5-1245-fr-en-train.json"


### /Data/shash/mul/finetuned_models_fr4_phi_32_ep5/checkpoint-1245

python -m src.compare_phi \
  --base_model "microsoft/Phi-4-mini-instruct" \
  --finetuned_model "/Data/shash/mul/finetuned_models_ja4_phi_32_ep5/checkpoint-1245" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "de" \
  --num_samples 250 \
  --subset "train" \
  --output_file "comparison_results_256/phi4/comparison_results_ja4_phi_32_ep5-1245-ja-de.json"


## Finetune Full SFT Qwen

Qwen/Qwen2.5-0.5B-Instruct

python -m src.finetune_qwen \
    --dataset_path /Data/shash/mul/hf_dataset2 \
    --language ja \
    --output_dir /Data/shash/mul/finetuned_models/qwen_ja_sft_fullprec_ep5/ \
    --wandb_project mulsum-qwen

5/1245 [01:05<4:25:46, 12.86s/it]
qwen-fr-full-finetune-ep5
Memory Usage:  14076MiB 
trainable params: 494,032,768 || all params: 494,032,768 || trainable%: 100.0000
wandb:                    epoch 4.98025
wandb:                eval/loss 0.7213
wandb:             eval/runtime 96.2833
wandb:  eval/samples_per_second 5.203
wandb:    eval/steps_per_second 5.203
wandb:            model/size_mb 1884.58545
wandb:       rouge/best_rouge-1 0.49959
wandb:       rouge/best_rouge-2 0.26626
wandb:       rouge/best_rouge-l 0.46354
wandb:        rouge/val_rouge-1 0.45137
wandb:        rouge/val_rouge-2 0.22606
wandb:        rouge/val_rouge-l 0.42444
wandb:               total_flos 6.913899205732147e+16
wandb:              train/epoch 4.98025
wandb:        train/global_step 1245
wandb:          train/grad_norm 1.24218
wandb:      train/learning_rate 0.0
wandb:               train/loss 0.0399
wandb:               train_loss 0.32778
wandb:            train_runtime 14974.6135
wandb: train_samples_per_second 1.335
wandb:   train_steps_per_second 0.083


### de
/Data/shash/mul/finetuned_models/qwen_de_sft_fullprec_ep5
13684MiB

### ja
/Data/shash/mul/finetuned_models/qwen_ja_sft_fullprec_ep5/checkpoint-1000

conda activate mul
## compare de
python -m src.compare_qwen \
  --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
  --finetuned_model "/Data/shash/mul/finetuned_models/qwen_ja_sft_fullprec_ep5/checkpoint-1000" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "en" \
  --num_samples 500 \
  --subset "train" \
  --output_file "comparison_results_256/qwen/comparison_results_ja4_qwen_sft_ep4-1000-ja-en-train.json"


## compare fr

python -m src.compare_qwen \
  --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
  --finetuned_model "/Data/shash/mul/finetuned_models/qwen_fr_sft_fullprec_ep5/checkpoint-1245" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "en" \
  --num_samples 250 \
  --subset "train" \
  --output_file "comparison_results_256/qwen/comparison_results_fr4_qwen_sft_ep5-1245-fr-en-train.json"

1st fr-fr-test 2/500 [00:20<1:24:22, 10.17s/it]
Average ROUGE Metrics                 
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Model                 ┃ ROUGE-1 ┃ ROUGE-2 ┃ ROUGE-L ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ Base Qwen Model       │ 0.2654  │ 0.0896  │ 0.2444  │
│ Fine-tuned Qwen Model │ 0.4800  │ 0.2704  │ 0.4518  │
└───────────────────────┴─────────┴─────────┴─────────┘


╭────────────────────────────────────────────── Fine-tuning Improvements ──────────────────────────────────────────────╮
│ ROUGE-1 Improvement: 80.86%                                                                                          ││ ROUGE-2 Improvement: 201.96%                                                                                         │
│ ROUGE-L Improvement: 84.81%  


2nd fr-de-test 24/500 [04:21<1:29:31, 11.29s/it]

3rd fr-en-test 6Model                 ┃ ROUGE-1 ┃ ROUGE-2 ┃ ROUGE-L ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ Base Qwen Model       │ 0.3146  │ 0.1178  │ 0.2872  │
│ Fine-tuned Qwen Model │ 0.3987  │ 0.1872  │ 0.3698  │
└───────────────────────┴─────────┴─────────┴─────────┘


╭────────────────────────────────────────────── Fine-tuning Improvements ──────────────────────────────────────────────╮
│ ROUGE-1 Improvement: 26.71%                                                                                          ││ ROUGE-2 Improvement: 58.99%                                                                                          │
│ ROUGE-L Improvement: 28.78% 
4th fr-fr-train 2/500 [00:24<1:39:00, 11.93s/it]
                 Average ROUGE Metrics                 
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Model                 ┃ ROUGE-1 ┃ ROUGE-2 ┃ ROUGE-L ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ Base Qwen Model       │ 0.2769  │ 0.0929  │ 0.2552  │
│ Fine-tuned Qwen Model │ 0.7404  │ 0.6563  │ 0.7312  │
└───────────────────────┴─────────┴─────────┴─────────┘


╭────────────────────────────────────────────── Fine-tuning Improvements ──────────────────────────────────────────────╮
│ ROUGE-1 Improvement: 167.40%                                                                                         ││ ROUGE-2 Improvement: 606.16%                                                                                         │
│ ROUGE-L Improvement: 186.46%  
5
                 Average ROUGE Metrics                 
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Model                 ┃ ROUGE-1 ┃ ROUGE-2 ┃ ROUGE-L ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ Base Qwen Model       │ 0.2464  │ 0.0652  │ 0.2292  │
│ Fine-tuned Qwen Model │ 0.3205  │ 0.1174  │ 0.3022  │
└───────────────────────┴─────────┴─────────┴─────────┘


╭────────────────────────────────────────────── Fine-tuning Improvements ──────────────────────────────────────────────╮
│ ROUGE-1 Improvement: 30.07%                                                                                          ││ ROUGE-2 Improvement: 79.85%                                                                                          │
│ ROUGE-L Improvement: 31.85%
6


## LLAMA 3.2 -1b Instruct with advanced techniques
pip install flash-attn --no-build-isolation

### fr
python -m src.finetune_llama \
  --model_name "meta-llama/Llama-3.2-1B-Instruct" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "fr" \
  --output_dir "/Data/shash/mul/finetuned_llama3-2-1b/llama3_1b_fr_ep5/" \
  --use_neftune \
  --use_flash_attention

wandb: Syncing run llama32-fr-r32-lr0.0005
  Setting up LoRA with r=32, alpha=64
INFO:__main__:Targeting all linear layers: ['k_proj', 'gate_proj', 'o_proj', 'up_proj', 'q_proj', 'v_proj', 'down_proj']
INFO:__main__:trainable params: 22,544,384 || all params: 1,258,358,784 || trainable%: 1.7916

7052MiB
| 7/1245 [00:55<2:39:10,  7.71s/it]

wandb: Run summary:
wandb:                    epoch 4.98025
wandb:                eval/loss 7.3836
wandb:             eval/runtime 162.1458
wandb:  eval/samples_per_second 3.09
wandb:    eval/steps_per_second 3.09
wandb:                eval_loss 7.3836
wandb:             eval_runtime 162.1458
wandb:  eval_samples_per_second 3.09
wandb:    eval_steps_per_second 3.09
wandb:            model_size_mb 1552.25781
wandb:       rouge/best_rouge-1 0.16996
wandb:       rouge/best_rouge-2 0.01082
wandb:       rouge/best_rouge-l 0.16703
wandb:      rouge/final_rouge-1 0.16553
wandb:      rouge/final_rouge-2 0.01189
wandb:      rouge/final_rouge-l 0.16059
wandb:        rouge/val_rouge-1 0.16942
wandb:        rouge/val_rouge-2 0.01054
wandb:        rouge/val_rouge-l 0.16195
wandb:               total_flos 1.8519637715081626e+17
wandb:              train/epoch 4.98025
wandb:      train/example_count 19920                                                                /run
wandb:        train/global_step 1245                                                                     s/iuzevoox
wandb:          train/grad_norm 0.76                                                               ma
wandb:      train/learning_rate 0.0
wandb:               train/loss 7.3644                                                            a
wandb:               train_loss 7.67968
wandb:            train_runtime 20082.4133                                                       l
wandb: train_samples_per_second 0.996                                                                7 133:57 13-Mar-25
wandb:   train_steps_per_second 0.062                                                           l
wandb:                                                                                             3-
wandb: 🚀 View run llama32-fr-r32-lr0.0005 at: https://wandb.ai/llm-summarization-fbt3r5/mulsum-llama/runs/xv67miu8  

/Data/shash/mul/finetuned_llama3-2-1b/llama3_1b_fr_ep5/checkpoint-1245



### de
python -m src.finetune_llama \
  --model_name "meta-llama/Llama-3.2-1B-Instruct" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "de" \
  --output_dir "/Data/shash/mul/finetuned_llama3-2-1b/llama3_1b_de_ep5/" \
  --use_neftune \
  --use_flash_attention

wandb: Syncing run llama32-de-r32-lr0.0005
INFO:__main__:trainable params: 22,544,384 || all params: 1,258,358,784 || trainable%: 1.7916

7164MiB

### ja
python -m src.finetune_llama \
  --model_name "meta-llama/Llama-3.2-1B-Instruct" \
  --dataset_path "/Data/shash/mul/hf_dataset2" \
  --language "ja" \
  --output_dir "/Data/shash/mul/finetuned_llama3-2-1b/llama3_1b_ja_ep5/" \
  --use_neftune \
  --use_flash_attention

9720MiB
2/1245 [00:27<4:45:26, 13.78s/it]


### LLM Evaluation


python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_de4_qwen_sft_ep5-1245-de-de-train.json --samples 20 --output llm_evaluator/de4_qwen_de_de-train.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_de4_qwen_sft_ep5-1245-de-fr-train.json --samples 20 --output llm_evaluator/de4_qwen_de_fr-train.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_de4_qwen_sft_ep5-1245-de-en-train.json --samples 20 --output llm_evaluator/de4_qwen_de_en-train.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_de4_qwen_sft_ep5-1245-de-de-test.json --samples 20 --output llm_evaluator/de4_qwen_de_de-test.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_de4_qwen_sft_ep5-1245-de-fr-test.json --samples 20 --output llm_evaluator/de4_qwen_de_fr-test.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_de4_qwen_sft_ep5-1245-de-en-test.json --samples 20 --output llm_evaluator/de4_qwen_de_en-test.json


python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_fr4_qwen_sft_ep5-1245-fr-de-train.json --samples 20 --output llm_evaluator/de4_qwen_fr_de-train.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_fr4_qwen_sft_ep5-1245-fr-fr-train.json --samples 20 --output llm_evaluator/de4_qwen_fr_fr-train.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_fr4_qwen_sft_ep5-1245-fr-en-train.json --samples 20 --output llm_evaluator/de4_qwen_fr_en-train.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_fr4_qwen_sft_ep5-1245-fr-de-test.json --samples 20 --output llm_evaluator/de4_qwen_fr_de-test.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_fr4_qwen_sft_ep5-1245-fr-fr-test.json --samples 20 --output llm_evaluator/de4_qwen_fr_fr-test.json

python -m src.llm_evaluator comparison_results_256/qwen/comparison_results_fr4_qwen_sft_ep5-1245-fr-en-test.json --samples 20 --output llm_evaluator/de4_qwen_fr_en-test.json
