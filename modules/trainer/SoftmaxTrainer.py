import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from typing import Any, Union, Dict
from ..models.embedder import SentenceEmbedder
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
from transformers import get_linear_schedule_with_warmup


class SoftMaxTrainer(pl.LightningModule):
    def __init__(
        self,
        embedder: SentenceEmbedder,
        num_labels: int,
        max_iters: int,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedder"])
        print(f"==>> self.hparams: {self.hparams}")
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
        loss = self.criterion(logits, labels.view(-1))
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        sentence_1, sentence_2, labels = batch
        sentence_1_embedding = self.embedder(sentence_1)
        sentence_2_embedding = self.embedder(sentence_2)
        diff = torch.abs(sentence_1_embedding - sentence_2_embedding)
        output = torch.cat([sentence_1_embedding, sentence_2_embedding, diff], dim=1)
        logits = self.classifier(output).argmax(dim=-1)
        acc = (logits == labels).float().mean()
        self.log("val_acc", acc)

    def test_step(
        self,
        batch,
        batch_idx,
    ):
        sentence_1, sentence_2, labels = batch
        sentence_1_embedding = self.embedder(sentence_1)
        sentence_2_embedding = self.embedder(sentence_2)
        diff = torch.abs(sentence_1_embedding - sentence_2_embedding)
        output = torch.cat([sentence_1_embedding, sentence_2_embedding, diff], dim=1)
        logits = self.classifier(output).argmax(dim=-1)
        acc = (logits == labels).float().mean()
        self.log("test_acc", acc)

    def configure_optimizers(self) -> Any:
        if self.hparams["optimizer_name"] == "Adam":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                **self.hparams["optimizer_hparams"],
            )

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams["num_warmup_steps"],
                num_training_steps=self.hparams["max_iters"],
            )

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            assert False, "Optimizer not supported"

    def _test_traning_step(self, inputs):
        sentence_1, sentence_2, labels = inputs
        sentence_1_embedding = self.embedder(sentence_1)
        sentence_2_embedding = self.embedder(sentence_2)
        diff = torch.abs(sentence_1_embedding - sentence_2_embedding)
        output = torch.cat([sentence_1_embedding, sentence_2_embedding, diff], dim=1)
        logits = self.classifier(output)
        if labels is not None:
            loss = self.criterion(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            return loss
        else:
            return output, logits
