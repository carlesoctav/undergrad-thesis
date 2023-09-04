import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Union
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(
        self,
        model_outputs: BaseModelOutputWithPoolingAndCrossAttentions,
        attention_mask: Tensor,
    ) -> Tensor:
        token_embeddings = model_outputs[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, features: Tensor) -> Tensor:
        features_normalized = F.normalize(features, p=2, dim=1)
        return features_normalized


class SentenceEmbedder(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        pooling_layer: nn.Module,
        normalize_layer: nn.Module,
        max_token: int = 512,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.pooling_layer = pooling_layer
        self.normalize_layer = normalize_layer
        self.max_token = max_token

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        outputs = self.model(**inputs)
        sentence_embedding = self.pooling_layer(outputs, inputs["attention_mask"])
        sentence_embedding = self.normalize_layer(sentence_embedding)

        return sentence_embedding

    def EmbedDimension(self):
        return self.model.config.hidden_size
