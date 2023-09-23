from modules.trainer.CosineTrainer import CosineTrainer
from modules.models.embedder import SentenceEmbedder
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers import (
    SentenceTransformer,
    SentencesDataset,
    InputExample,
    losses,
)


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

model_embedder = SentenceEmbedder(
    pretrained_model="sentence-transformers/all-MiniLM-L6-v2",
    pooling_layer="mean",
    normalize_layer="l2",
)

model_trainer = CosineTrainer(
    embedder=model_embedder,
    max_iters=100,
)

sentence_1 = ["This is an example sentence", "Each sentence is converted"]
sentence_2 = ["Each sentence is converted", "This is an example sentence"]
labels = Tensor([1.0, 0.0])
sentence_1 = tokenizer(sentence_1, padding=True, truncation=True, return_tensors="pt")
print(f"==>> sentence_1: {sentence_1}")
sentence_2 = tokenizer(sentence_2, padding=True, truncation=True, return_tensors="pt")
print(f"==>> sentence_2: {sentence_2}")
model_trainer.eval()

model_sentence_transformer = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

cosine_loss_st = CosineSimilarityLoss(model_sentence_transformer)
cosine_loss_st.eval()

with torch.no_grad():
    output_1 = model_trainer._test_traning_step((sentence_1, sentence_2, labels))
    output_2 = cosine_loss_st((sentence_1, sentence_2), labels)


print(f"==>> output_1: {output_1}")
print(f"==>> output_2: {output_2}")
