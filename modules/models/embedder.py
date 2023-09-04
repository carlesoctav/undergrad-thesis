from typing import Dict, Literal, Union

import torch.nn as nn

from torch import Tensor
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.tokenization_utils_base import BatchEncoding

from .pooling import ClsPooling, MeanPooling


class SentenceEmbedder(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        pooling_layer: Literal["mean", "cls"],
        normalize_layer: Literal["l2", None] = None,
    ):
        """
        Class to embed sentences using pretrained Transformer model from HuggingFace
        Args:
            pretrained_model (str): pretrained Transformer model from HuggingFace
            pooling_layer (Literal["mean", "cls"]): pooling layer to use
            normalize_layer (Literal["l2", "l1","None"]): normalization layer to use

        """

        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.embedder_metadata = {
            "pretrained_model": pretrained_model,
            "pooling_layer": pooling_layer,
            "normalize_layer": normalize_layer,
            "pretrained_model_config": self.model.config,
        }

        if pooling_layer == "mean":
            self.pooling_layer = MeanPooling()
        elif pooling_layer == "cls":
            self.pooling_layer = ClsPooling()

        if normalize_layer == "l2":
            self.normalize_layer = nn.LayerNorm(
                self.model.config.hidden_size, eps=1e-12
            )
        elif normalize_layer == None:
            self.normalize_layer = nn.Identity()

    def forward(
        self,
        inputs: Union[Dict[str, Tensor], BatchEncoding],
    ) -> Tensor:
        outputs = self.model(**inputs)
        sentence_embedding = self.pooling_layer(outputs, inputs["attention_mask"])
        sentence_embedding = self.normalize_layer(sentence_embedding)

        return sentence_embedding
