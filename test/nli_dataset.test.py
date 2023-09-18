from modules.datasets.nli_dataset import NLIDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def collate_fn(batch):
    sentence_1 = [item[0] for item in batch]
    sentence_2 = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    sentence_1 = tokenizer(
        sentence_1, padding=True, truncation=True, return_tensors="pt"
    )
    sentence_2 = tokenizer(
        sentence_2, padding=True, truncation=True, return_tensors="pt"
    )
    return (
        sentence_1,
        sentence_2,
        torch.tensor([item[2] for item in batch]),
    )


path = "./data/carles-undergrad-thesis/indo-snli_train.json"

dataset = NLIDataset(path)
print(f"==>> dataset[0]: {dataset[0]}")
data_loader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)

data = next(iter(data_loader))
print(f"==>> data: {data}")
