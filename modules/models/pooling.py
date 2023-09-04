from typing import Dict, List, Tuple, Union
from torch import Tensor
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        model_outputs: BaseModelOutputWithPoolingAndCrossAttentions,
        attention_mask: Tensor,
    ) -> Tensor:
        token_embeddings = model_outputs[0]

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(
            token_embeddings * input_mask_expanded,
            1,
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ClsPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, model_outputs: BaseModelOutputWithPoolingAndCrossAttentions
    ) -> Tensor:
        return model_outputs.last_hidden_state[:, 0]
