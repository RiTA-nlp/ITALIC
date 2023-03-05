import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import argparse
from datasets import load_dataset
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-multilingual-uncased')
    parser.add_argument('--dataset_name_or_path', type=str, default='RiTA-nlp/italic-easy')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--max_input_length', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument('--output_dir', type=str, default='models/')
    return parser.parse_args()

args = parse_args()

dataset = load_dataset(args.dataset_name_or_path)

train_sentences = dataset["train"]["utt"]
train_labels = dataset["train"]["intent"]

val_sentences = dataset["validation"]["utt"]
val_labels = dataset["validation"]["intent"]

# find the number of unique labels
unique_labels = set(train_labels)
num_labels = len(unique_labels)
print("Number of unique labels:", num_labels)

# map labels to integers
# order labels alphabetically
label_to_int = {label: i for i, label in enumerate(sorted(unique_labels))}
int_to_label = {i: label for label, i in label_to_int.items()}
train_labels = [label_to_int[label] for label in train_labels]
val_labels = [label_to_int[label] for label in val_labels]

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
safe_model_name = args.model_name_or_path.replace("/", "-")

class IntentClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_name_or_path: str,
        max_input_length: int = 64,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.texts = texts
        self.labels = labels
        self.max_input_length = max_input_length
        self.encodings = self.tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt",
        )


    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IntentClassificationDataset(
    train_sentences,
    train_labels,
    args.model_name_or_path,
    args.max_input_length,
)

val_dataset = IntentClassificationDataset(
    val_sentences,
    val_labels,
    args.model_name_or_path,
    args.max_input_length,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    try:
        predictions = np.argmax(predictions, axis=1)
    except Exception:
        print("predictions[0] shape:", predictions[0].shape)
        predictions = np.argmax(predictions[0], axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir=args.output_dir + safe_model_name,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=max(1, args.batch_size // 2),
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

model = trainer.model

model.save_pretrained(args.output_dir + safe_model_name + "/best_model/")
tokenizer.save_pretrained(args.output_dir + safe_model_name + "/best_model/")

'''
python ft_text.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --dataset_name_or_path RiTA-nlp/italic-easy \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --max_input_length 64 \
    --gradient_accumulation_steps 1 \
    --output_dir text_models/
'''