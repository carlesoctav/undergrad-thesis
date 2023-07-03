from BaseModel import SentenceEmbedder, MeanPooling, Normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Union
from torch import Tensor

class SoftmaxTrainer(pl.LightningModule):
    def __init__(self, 
                 pretrained_model:str, 
                 pooling_layer:nn.Module, 
                 normalize_layer:nn.Module,
                 max_token: int = 512,
                 learning_rate: float = 0.0001,
                 weight_decay: float = 0.0001,
                 embed_dim: int = 768,
                 num_labels: int = 2,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SentenceEmbedder(pretrained_model, pooling_layer, normalize_layer, max_token)
        self.dim = self.model.EmbedDimension()
        self.classifier = nn.Linear(3*self.dim, num_labels)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self,
                inputs: Dict[str, Tensor]):
        return self.model(inputs)
    
    def training_step(self, 
                      batch, 
                      batch_idx):
        
        sentence_1, sentence_2, labels = batch
        sentence_1_embedding = self(sentence_1)
        sentence_2_embedding = self(sentence_2)
        diff = torch.abs(sentence_1_embedding - sentence_2_embedding)
        output = torch.cat([sentence_1_embedding, sentence_2_embedding, diff], dim=1)
        logits = self.classifier(output)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        sentence_1, sentence_2, labels = batch
        sentence_1_embedding = self(sentence_1)
        sentence_2_embedding = self(sentence_2)
        diff = torch.abs(sentence_1_embedding - sentence_2_embedding)
        output = torch.cat([sentence_1_embedding, sentence_2_embedding, diff], dim=1)
        logits = self.classifier(output)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer