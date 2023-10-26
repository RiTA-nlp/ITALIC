from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight

""" Trainer Class """
class WeightedTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


""" Define training arguments """ 
def define_training_args(output_dir, batch_size, num_epochs=30, gradient_accumulation_steps=1):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",  
        save_strategy = "epoch",        
        learning_rate=1.0e-4,           
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,    
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_steps=100,                 
        save_steps=100,                 
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        fp16=True,
        fp16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        )
    return training_args
    

""" Define Metric """
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    print(f"Accuracy: {acc*100:.3f}")
    print(f"F1: {f1*100:.3f}")
    return { 'accuracy': acc, 'f1': f1 }
