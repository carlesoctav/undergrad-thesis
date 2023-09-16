from modules.trainer.SoftmaxTrainer import SoftMaxTrainer
from modules.models.embedder import SentenceEmbedder
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import SoftmaxLoss


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

model_embedder = SentenceEmbedder(
    pretrained_model="sentence-transformers/all-MiniLM-L6-v2",
    pooling_layer="mean",
    normalize_layer="l2",
)
model_trainer = SoftMaxTrainer(
    embedder=model_embedder,
    num_labels=2,
)


sentence_1 = ["This is an example sentence", "Each sentence is converted"]
sentence_2 = ["Each sentence is converted", "This is an example sentence"]
sentence_1 = tokenizer(sentence_1, padding=True, truncation=True, return_tensors="pt")
print(f"==>> sentence_1: {sentence_1}")
sentence_2 = tokenizer(sentence_2, padding=True, truncation=True, return_tensors="pt")
print(f"==>> sentence_2: {sentence_2}")
model_trainer.eval()


model_sentence_transformer = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

softmax_loss = SoftmaxLoss(
    model_sentence_transformer, sentence_embedding_dimension=384, num_labels=2
)

softmax_loss.eval()

with torch.no_grad():
    output_1 = model_trainer._test_traning_step((sentence_1, sentence_2))
    output_2 = softmax_loss((sentence_1, sentence_2), None)

print(f"==>> output_1.shape: {output_1[0].shape}")
print(f"==>> output_2.shape: {output_2[0].shape}")

print(f"==>> output_1: {output_1}")
print(f"==>> output_2: {output_2}")
