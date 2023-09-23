import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from typing import Any, Union, Dict, Literal
from ..models.embedder import SentenceEmbedder
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
from transformers import get_linear_schedule_with_warmup
from enum import Enum


class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """

    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class TripletTrainer(pl.LightningModule):
    def __init__(
        self,
        embedder: SentenceEmbedder,
        max_iters: int,
        triplet_margin: float = 2.0,
        distance_metric: Literal["COSINE", "EUCLIDEAN", "MANHATTAN"] = "EUCLIDEAN",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedder"])
        print(f"==>> self.hparams: {self.hparams}")
        self.embedder = embedder

        if distance_metric == "COSINE":
            self.distance_metric = TripletDistanceMetric.COSINE
        elif distance_metric == "EUCLIDEAN":
            self.distance_metric = TripletDistanceMetric.EUCLIDEAN
        elif distance_metric == "MANHATTAN":
            self.distance_metric = TripletDistanceMetric.MANHATTAN
        else:
            assert False, "Distance metric not supported"

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
        sentence_a, sentence_p, sentence_n = batch
        sentence_a_embedding = self.embedder(sentence_a)
        sentence_p_embedding = self.embedder(sentence_p)
        sentence_n_embedding = self.embedder(sentence_n)

        distance_pos = self.distance_metric(sentence_a_embedding, sentence_p_embedding)
        distance_neg = self.distance_metric(sentence_a_embedding, sentence_n_embedding)
        loss = F.relu(distance_pos - distance_neg + self.hparams["triplet_margin"])
        self.log("train_loss", loss.mean(), on_step=False, on_epoch=True)
        return loss.mean()

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        sentence_a, sentence_p, sentence_n = batch
        sentence_a_embedding = self.embedder(sentence_a)
        sentence_p_embedding = self.embedder(sentence_p)
        sentence_n_embedding = self.embedder(sentence_n)

        distance_pos = self.distance_metric(sentence_a_embedding, sentence_p_embedding)
        distance_neg = self.distance_metric(sentence_a_embedding, sentence_n_embedding)
        loss = F.relu(distance_pos - distance_neg + self.hparams["triplet_margin"])
        self.log("val_loss", loss.mean(), on_step=False, on_epoch=True)

    def test_step(
        self,
        batch,
        batch_idx,
    ):
        sentence_a, sentence_p, sentence_n = batch
        sentence_a_embedding = self.embedder(sentence_a)
        sentence_p_embedding = self.embedder(sentence_p)
        sentence_n_embedding = self.embedder(sentence_n)

        distance_pos = self.distance_metric(sentence_a_embedding, sentence_p_embedding)
        distance_neg = self.distance_metric(sentence_a_embedding, sentence_n_embedding)
        loss = F.relu(distance_pos - distance_neg + self.hparams["triplet_margin"])
        self.log("test_loss", loss.mean(), on_step=False, on_epoch=True)

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
        sentence_a, sentence_p, sentence_n = inputs
        sentence_a_embedding = self.embedder(sentence_a)
        sentence_p_embedding = self.embedder(sentence_p)
        sentence_n_embedding = self.embedder(sentence_n)

        distance_pos = self.distance_metric(sentence_a_embedding, sentence_p_embedding)
        distance_neg = self.distance_metric(sentence_a_embedding, sentence_n_embedding)

        losses = F.relu(distance_pos - distance_neg + self.hparams["triplet_margin"])
        return losses.mean()
