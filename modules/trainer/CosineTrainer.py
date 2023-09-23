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


class CosineTrainer(pl.LightningModule):
    def __init__(
        self,
        embedder: SentenceEmbedder,
        max_iters: int,
        loss_fct: nn.Module = nn.MSELoss(),
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedder", "loss_fct"])
        print(f"==>> self.hparams: {self.hparams}")
        self.embedder = embedder
        self.lost_fct = loss_fct

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
        output = torch.cosine_similarity(sentence_1_embedding, sentence_2_embedding)
        loss = self.lost_fct(output, labels.view(-1))
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
        output = torch.cosine_similarity(sentence_1_embedding, sentence_2_embedding)
        loss = self.lost_fct(output, labels.view(-1))
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(
        self,
        batch,
        batch_idx,
    ):
        sentence_1, sentence_2, labels = batch
        sentence_1_embedding = self.embedder(sentence_1)
        sentence_2_embedding = self.embedder(sentence_2)
        output = torch.cosine_similarity(sentence_1_embedding, sentence_2_embedding)
        loss = self.lost_fct(output, labels.view(-1))
        self.log("test_loss", loss, on_step=False, on_epoch=True)

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
        output = torch.cosine_similarity(sentence_1_embedding, sentence_2_embedding)
        loss = self.lost_fct(output, labels.view(-1))
        return loss
