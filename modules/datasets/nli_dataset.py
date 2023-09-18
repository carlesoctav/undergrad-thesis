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
        self.label = []
        self.metadata = {
            "path": path,
        }

        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

        for datum in self.data:
            self.sentence_1.append(datum["sentence_1"])
            self.sentence_2.append(datum["sentence_2"])
            self.label.append(datum["label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return [
            self.sentence_1[idx],
            self.sentence_2[idx],
            self.label[idx],
        ]
