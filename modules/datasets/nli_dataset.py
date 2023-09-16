import torch
from torch.utils.data import DataLoader, Dataset
import json
from collections import OrderedDict


class NLIDataset(Dataset):
    def __init__(
        self,
        path: str,
    ):
        self.sentence_1 = []
        self.sentence_2 = []
        self.labels = []
        self.metadata = {
            "path": path,
        }

        with open(path, "r") as f:
            self.data = json.load(f)

        for datum in self.data:
            self.sentence_1.append(datum["sentence_1"])
            self.sentence_2.append(datum["sentence_2"])
            self.labels.append(datum["label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return [
            self.sentence_1[idx],
            self.sentence_2[idx],
            self.labels[idx],
        ]
