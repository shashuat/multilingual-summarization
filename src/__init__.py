import os

os.environ["HF_HOME"] = "/Data/shash/.cache"
os.environ["HF_HUB_CACHE"] = "/Data/shash/.cache/hub"

print(os.environ["HF_HOME"])

from dotenv import load_dotenv
import wandb

from huggingface_hub import login

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
wandb_token = os.getenv("WANDB_TOKEN")

if hf_token:
    login(hf_token)
    print("Logged in to Hugging Face successfully!")
else:
    print("HF_TOKEN not found in .env file.")

if wandb_token:
    wandb.login(key=wandb_token)
    print("Logged in to Weights & Biases successfully!")
else:
    print("WANDB_TOKEN not found in .env file.")
