
# train.py
from transformers import Trainer, TrainingArguments
import evaluate
from typing import Dict
from config import TrainingConfig, ModelConfig
import os

class SummaryTrainer:
    def __init__(
        self,
        model: BaseSummarizer,
        train_config: TrainingConfig,
        model_config: ModelConfig
    ):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        self.rouge = evaluate.load('rouge')
        self.bert_score = evaluate.load('bertscore')
        
    def train(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir=self.train_config.output_dir,
            learning_rate=self.model_config.learning_rate,
            num_train_epochs=self.model_config.num_epochs,
            per_device_train_batch_size=self.model_config.train_batch_size,
            per_device_eval_batch_size=self.model_config.eval_batch_size,
            warmup_steps=self.model_config.warmup_steps,
            weight_decay=self.model_config.weight_decay,
            logging_dir=self.train_config.logging_dir,
            logging_steps=self.train_config.logging_steps,
            evaluation_strategy=self.train_config.evaluation_strategy,
            save_strategy=self.train_config.save_strategy,
            load_best_model_at_end=self.train_config.load_best_model_at_end,
            gradient_accumulation_steps=self.model_config.gradient_accumulation_steps
        )
        
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()
        
    def evaluate(self, test_dataset) -> Dict:
        predictions = []
        references = []
        
        for batch in test_dataset:
            pred = self.model.generate_summary(batch["text"])
            predictions.append(pred)
            references.append(batch["summary"])
            
        metrics = {
            "rouge": self.rouge.compute(
                predictions=predictions,
                references=references
            ),
            "bert_score": self.bert_score.compute(
                predictions=predictions,
                references=references,
                lang="multi"
            )
        }
        
        return metrics
