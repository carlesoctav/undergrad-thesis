from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('carles-undergrad-thesis/st-indo-bert-mmarco-knowlegde-distillation-256')
print(model.max_seq_length)

model.max_seq_length = 256
print(model.max_seq_length)

# model.save_to_hub(repo_name="st-indo-bert-mmarco-knowlegde-distillation-256", organization="carles-undergrad-thesis")