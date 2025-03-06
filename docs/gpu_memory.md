# **Summary: Attempts to Reduce GPU Memory Usage**

## **Memory Optimization Methods and Results**

| Method | Result |
| --- | --- |
| 4-bit Quantization (`load_in_4bit=True`) + batch size 1 | 20GB VRAM |
| 4-bit Quantization + batch size 1 + Flash Attention (`flash_attention_2`) | 20GB VRAM |
| Use `max_memory` settings | Encountered error: "All tensors should be in the same device" |
| Manual CPU offload | Encountered error: "All tensors should be in the same device" |

---

# **Flash Attention Implementation**

### **Installation in Google Colab**
To install Flash Attention in Google Colab, run the following command:
```bash
!pip install flash-attn --no-build-isolation
```

### **Execution Command in Google Colab**
Run the script to generate summaries:
```bash
!python /content/drive/MyDrive/text_mining/multilingual-summarization/src/generate_summaries.py \
    --raw-data-dir /content/drive/MyDrive/text_mining/multilingual-summarization/data/raw \
    --summaries-dir /content/drive/MyDrive/text_mining/multilingual-summarization/data/summaries_Qwen2_5_32B \
    --languages ja \
    --model-name Qwen/Qwen2.5-32B-Instruct
```

### **Modification of Model Settings**
Modify `generate_summaries.py` to enable Flash Attention:
```python
# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
```

### **Results**
```
GPU RAM: 20.1 / 22.5 GB
```

---

# **CPU Offload Implementation**

### **Configuration for CPU Offload**
Modify `generate_summaries.py` to configure CPU offloading:
```python
# Configure quantization
quantization_config = None
if torch.cuda.is_available():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
```

Modify `generate_summaries.py` to load the model with CPU offloading:
```python
# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=DEVICE_MAP,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)
```

Here, DEVICE_MAP is a dictionary of `src/device_map.json`.

### **Execution Command in Google Colab**
Run the script to generate summaries with CPU offloading:
```bash
!python /content/drive/MyDrive/text_mining/multilingual-summarization/src/generate_summaries.py \
    --raw-data-dir /content/drive/MyDrive/text_mining/multilingual-summarization/data/raw \
    --summaries-dir /content/drive/MyDrive/text_mining/multilingual-summarization/data/summaries_Qwen2_5_32B \
    --languages ja \
    --model-name Qwen/Qwen2.5-32B-Instruct
```

### **Results**
By setting **layers up to 30 for CPU** and keeping the remaining layers on **GPU**, the GPU memory usage is reduced significantly:
```
System RAM: 31.9 / 53.0 GB
GPU RAM: 9.8 / 22.5 GB
```

However, the following error was encountered during execution:
```
Generating ja summaries (batch 46):   0% 0/1 [00:00<?, ?it/s]
Error generating summary for article a4c4ea472e in ja:
All input tensors need to be on the same GPU, but found some tensors to not be on a GPU.
```

