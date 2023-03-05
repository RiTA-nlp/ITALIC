import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from datasets import load_dataset
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
# classification_report
from sklearn.metrics import classification_report
from tqdm import tqdm
# suppress all warnings
import warnings
warnings.filterwarnings("ignore")
import os

# set environment variable TOKENIZERS_PARALLELISM to false in order to prevent tokenizers from using all available CPU cores
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-multilingual-uncased')
    parser.add_argument('--dataset_name_or_path', type=str, default='RiTA-nlp/italic-easy')
    parser.add_argument('--split_name', type=str, default='test')
    parser.add_argument('--max_input_length', type=int, default=64)
    parser.add_argument('--cuda', action='store_true')
    return parser.parse_args()

args = parse_args()

dataset = load_dataset(args.dataset_name_or_path)

test_sentences = dataset[args.split_name]["utt"]
test_labels = dataset[args.split_name]["intent"]
train_labels = dataset["train"]["intent"]

# find the number of unique labels
unique_labels = set(train_labels)
num_labels = len(unique_labels)
print ("Number of unique labels:", num_labels)

# map labels to integers
# order labels alphabetically
label_to_int = {label: i for i, label in enumerate(sorted(unique_labels))}
int_to_label = {i: label for label, i in label_to_int.items()}
test_labels = [label_to_int[label] for label in test_labels]

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
if args.cuda:
    model = model.cuda()


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


test_dataset = IntentClassificationDataset(
    test_sentences,
    test_labels,
    args.model_name_or_path,
    args.max_input_length,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    try:
        predictions = np.argmax(predictions, axis=1)
    except Exception:
        print("predictions[0] shape:", predictions[0].shape)
        predictions = np.argmax(predictions[0], axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# evaluate on test set
pred_classes = []
true_classes = []

print(f"Evaluating model {args.model_name_or_path} on dataset {args.dataset_name_or_path} ({args.split_name} split)")

with torch.no_grad():
    for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Evaluating"):
        if args.cuda:
            batch = {key: val.cuda() for key, val in batch.items()}
            
        predictions = model(**batch)
        try:
            pred_classes.extend(torch.argmax(predictions.logits, dim=1).tolist())
        except Exception:
            pred_classes.extend(torch.argmax(predictions[0], dim=1).tolist())
        
        true_classes.extend(batch["labels"].tolist())

# print 2 digits after the decimal point
print(f"Accuracy: {accuracy_score(true_classes, pred_classes)*100:.2f}")
print(f"F1: {f1_score(true_classes, pred_classes, average='macro')*100:.2f}")
# print("Classification report:")
# print(classification_report(true_classes, pred_classes))
print ("\n\n")

'''
python eval_text.py \
    --model_name_or_path text_models/easy/dbmdz-bert-base-italian-xxl-uncased/best_model/ \
    --dataset_name_or_path RiTA-nlp/italic-easy \
    --split_name test \
    --max_input_length 64
    --cuda
'''