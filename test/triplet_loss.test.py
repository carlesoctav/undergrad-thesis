from modules.trainer.TripletTrainer import TripletTrainer
from modules.models.embedder import SentenceEmbedder
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import TripletLoss
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

model_trainer = TripletTrainer(
    embedder=model_embedder,
    max_iters=100,
    triplet_margin=5.0,
)

sentence_a = ["This is an example sentence", "Each sentence is converted"]
sentence_p = ["Each sentence is converted", "This is an example sentence"]
sentence_n = ["hello", "world"]


sentence_a = tokenizer(sentence_a, padding=True, truncation=True, return_tensors="pt")
print(f"==>> sentence_a: {sentence_a}")
sentence_p = tokenizer(sentence_p, padding=True, truncation=True, return_tensors="pt")
print(f"==>> sentence_p: {sentence_p}")
sentence_n = tokenizer(sentence_n, padding=True, truncation=True, return_tensors="pt")
print(f"==>> sentence_n: {sentence_n}")

model_sentence_transformer = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

triplet_loss_st = TripletLoss(model_sentence_transformer)
triplet_loss_st.eval()

with torch.no_grad():
    output_1 = model_trainer._test_traning_step((sentence_a, sentence_p, sentence_n))
    output_2 = triplet_loss_st((sentence_a, sentence_p, sentence_n), None)


print(f"==>> output_1: {output_1}")
print(f"==>> output_2: {output_2}")
