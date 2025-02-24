# models.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
from auto_gptq import AutoGPTQForCausalLM
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Optional
from config import ModelConfig

class BaseSummarizer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
    def _init_tokenizer(self):
        raise NotImplementedError
        
    def _init_model(self):
        raise NotImplementedError
        
    def generate_summary(self, text: str) -> str:
        raise NotImplementedError

class MT5Summarizer(BaseSummarizer):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._init_tokenizer()
        self._init_model()
        
    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        
    def _init_model(self):
        base_model = AutoModelForSeq2SeqGeneration.from_pretrained(
            self.config.base_model_name
        ).to(self.device)
        
        if not hasattr(self, 'is_finetuned'):
            # Add LoRA for fine-tuning
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            self.model = get_peft_model(base_model, lora_config)
        else:
            self.model = base_model
            
    def generate_summary(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=self.config.max_summary_length,
            num_beams=self.config.num_beams,
            length_penalty=self.config.length_penalty,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class QuantizedQwenSummarizer(BaseSummarizer):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._init_tokenizer()
        self._init_model()
        
    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.summary_model_name)
        
    def _init_model(self):
        if self.config.use_quantization:
            self.model = AutoGPTQForCausalLM.from_quantized(
                self.config.summary_model_name,
                use_safetensors=True,
                device_map="auto",
                use_triton=False
            )
        else:
            self.model = AutoModelForSeq2SeqGeneration.from_pretrained(
                self.config.summary_model_name
            ).to(self.device)
            
    def generate_summary(self, text: str) -> str:
        prompt = f"Summarize the following text concisely:\n\n{text}\n\nSummary:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_summary_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
