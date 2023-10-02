from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("carlesoctav/indo-sentence-bert-KD")

model.save_to_hub(
    repo_name="st-indo-bert-mmarco-knowlegde-distillation",
    organization="carles-undergrad-thesis",
)
