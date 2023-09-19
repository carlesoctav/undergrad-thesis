from modules.models.embedder import SentenceEmbedder
from modules.trainer.SoftmaxTrainer import SoftMaxTrainer
from experiments.nli_model.nli_dataset import NLIDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import pytorch_lightning as pl
from tqdm import tqdm


def batch_tokenize(batch):
    global tokenizer

    sentence_1 = []
    sentence_2 = []
    labels = []

    for item in batch:
        sentence_1.append(item[0])
        sentence_2.append(item[1])
        labels.append(item[2])

    sentence_1 = tokenizer(
        sentence_1, padding=True, truncation=True, return_tensors="pt"
    )
    sentence_2 = tokenizer(
        sentence_2, padding=True, truncation=True, return_tensors="pt"
    )

    labels = torch.tensor(labels, dtype=torch.long)
    return sentence_1, sentence_2, labels


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceEmbedder(
    pretrained_model="sentence-transformers/all-MiniLM-L6-v2",
    pooling_layer="mean",
    normalize_layer=None,
)

embedder_trainer = SoftMaxTrainer(embedder, num_labels=3)
dataset = NLIDataset("./data/carles-undergrad-thesis/indo-snli_test.json")
data_loader = DataLoader(
    dataset, batch_size=128, collate_fn=batch_tokenize, num_workers=4
)

# embedder_trainer.to("cuda")


# for batch in tqdm(data_loader):
#     batch = [item.to("cuda") for item in batch]
#     print(f"==>> batch: {batch}")

#     embedder_trainer._test_traning_step(batch)

trainer = pl.Trainer(accelerator="gpu", max_epochs=2, logger=True)

trainer.fit(embedder_trainer, data_loader)
