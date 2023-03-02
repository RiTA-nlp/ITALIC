# import comet_ml
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

""" Trainer Class """
class WeightedTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


""" Define training arguments """ 
# def define_training_args(output_dir, batch_size, num_steps): 
def define_training_args(output_dir, batch_size, num_epochs, gradient_accumulation_steps=1):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",  # evaluation_strategy = "steps",
        save_strategy = "epoch",        # save_strategy = "steps",
        learning_rate=1.0e-4,           # learning_rate=3e-5,
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


# """ Define Class Weights """
# def compute_class_weights(ds_train, label2id):
#     class_weights = class_weight.compute_class_weight(
#         'balanced',
#         classes = np.unique(ds_train["intent"]),
#         Y = np.array(label2id[el for el in ds_train["intent"]]),
#     )
#     class_weights = torch.tensor(class_weights, device="cuda", dtype=torch.float32)
#     return class_weights


""" Define Metric """
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    print('Accuracy: ' + str(acc))
    return { 'accuracy': acc }