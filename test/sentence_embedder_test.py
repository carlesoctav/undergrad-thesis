from modules.models.embedder import SentenceEmbedder
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


model_embedder = SentenceEmbedder(
    pretrained_model="sentence-transformers/all-MiniLM-L6-v2",
    pooling_layer="mean",
    normalize_layer="l2",
)

print(model_embedder.embedder_metadata["pretrained_model_config"].hidden_size)


# Sentences we want sentence embeddings for
sentences = ["This is an example sentence", "Each sentence is converted"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    model_output_1 = model_embedder(encoded_input)

sentence_embeddings_1 = mean_pooling(model_output, encoded_input["attention_mask"])
sentence_embeddings_1 = F.normalize(sentence_embeddings_1, p=2, dim=1)
sentence_embeddings_2 = model_output_1
print(f"==>> sentence_embeddings_2.shape: {sentence_embeddings_2.shape}")

assert torch.allclose(sentence_embeddings_1, sentence_embeddings_2)
