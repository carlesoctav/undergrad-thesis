from modules.models.embedder import SentenceEmbedder
from modules.trainer.SoftmaxTrainer import SoftMaxTrainer
from modules.datasets.nli_dataset import NLIDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import pytorch_lightning as pl


def batch_tokenize(batch):
    global tokenizer
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


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceEmbedder(
    pretrained_model="sentence-transformers/all-MiniLM-L6-v2",
    pooling_layer="mean",
    normalize_layer=None,
)
embedder_trainer = SoftMaxTrainer(embedder, num_labels=3)
dataset = NLIDataset("./data/dummy_data.jsonl")
data_loader = DataLoader(dataset, batch_size=2, collate_fn=batch_tokenize)

for data in data_loader:
    print(f"==>> data: {data}")
    a = embedder_trainer._test_traning_step(data)
    print(f"==>> a: {a}")

trainer = pl.Trainer(accelerator="cpu", max_epochs=10)

trainer.fit(embedder_trainer, data_loader)
