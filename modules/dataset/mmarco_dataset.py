import torch
from torch.utils.data import DataLoader, Dataset
import json
from collections import OrderedDict


class MmarcoDataset(Dataset):
    def __init__(
        self,
        path: str,
    ):
        self.sentence_1 = []
        self.sentence_2 = []
        self.sentence_3 = []
        self.metadata = {
            "path": path,
        }

        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

        for datum in self.data:
            if datum["query"] and datum["positive"] and datum["negative"]:
                self.sentence_1.append(datum["query"].strip())
                self.sentence_2.append(datum["positive"].strip())
                self.sentence_3.append(datum["negative"].strip())

    def __len__(self):
        return len(self.sentence_1)

    def __getitem__(self, idx):
        return (
            self.sentence_1[idx],
            self.sentence_2[idx],
            self.sentence_3[idx],
        )
