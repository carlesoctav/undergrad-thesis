import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from typing import Any, Union, Dict
from ..models.embedder import SentenceEmbedder
from transformers.tokenization_utils_base import BatchEncoding


class SoftMaxTrainer(pl.LightningModule):
    def __init__(
        self,
        embedder: SentenceEmbedder,
        num_labels: int,
    ):
        super().__init__()
        self.embedder = embedder
        self.dim = self.embedder.embedder_metadata[
            "pretrained_model_config"
        ].hidden_size
        self.classifier = nn.Linear(3 * self.dim, num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        inputs: Union[Dict[str, Tensor], BatchEncoding],
    ) -> Tensor:
        outputs = self.embedder(**inputs)
        return outputs

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        sentence_1, sentence_2, labels = batch
        sentence_1_embedding = self.embedder(sentence_1)
        sentence_2_embedding = self.embedder(sentence_2)
        diff = torch.abs(sentence_1_embedding - sentence_2_embedding)
        output = torch.cat([sentence_1_embedding, sentence_2_embedding, diff], dim=1)
        logits = self.classifier(output)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _test_traning_step(self, inputs):
        sentence_1, sentence_2, labels = inputs
        sentence_1_embedding = self.embedder(sentence_1)
        sentence_2_embedding = self.embedder(sentence_2)
        diff = torch.abs(sentence_1_embedding - sentence_2_embedding)
        output = torch.cat([sentence_1_embedding, sentence_2_embedding, diff], dim=1)
        logits = self.classifier(output)

        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss
        else:
            return output, logits
