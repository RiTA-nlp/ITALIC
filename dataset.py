import torch
import torchaudio
import librosa
import numpy as np


""" Dataset Class """
class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, feature_extractor, max_duration, device):
        self.examples = examples['path']
        self.labels = examples['label']
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.device = device 

    def __getitem__(self, idx):
        inputs = self.feature_extractor(
            librosa.resample(
                np.asarray(torchaudio.load(self.examples[idx])[0]), 48_000, 16_000
                ).squeeze(0),
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


