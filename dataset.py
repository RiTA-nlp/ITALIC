import torch
import torchaudio
import librosa
import numpy as np


""" Dataset Class """
class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, feature_extractor, label2id, max_duration, device):
        self.examples = examples
        self.labels = [int(label2id[e]) for e in examples['intent']]
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.device = device 

    def __getitem__(self, idx):
        inputs = self.feature_extractor(
            self.examples[idx]['audio']['array'],
            sampling_rate=self.feature_extractor.sampling_rate, 
            return_tensors="pt",
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
            truncation=True,
            padding='max_length'
        )
        item = {'input_values': inputs['input_values'].squeeze(0)}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.examples)


