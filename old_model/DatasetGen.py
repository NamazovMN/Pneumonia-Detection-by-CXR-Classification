import torch
import numpy as np
from torch.utils.data import Dataset


class Xrays(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data, self.labels = self.generate_dataset(dataset['data'], dataset['labels'])

    def generate_dataset(self, data, labels):
        data = torch.FloatTensor(data)
        labels = torch.LongTensor(labels)
        return data, labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'labels': self.labels[idx]
        }
