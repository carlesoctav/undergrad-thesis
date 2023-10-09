from datasets import load_dataset

dataset = load_dataset("carles-undergrad-thesis/indo-mmarco-500k")
dataset = dataset["train"]

print(f"==>> dataset: {dataset}")

dataset.to_json(
    "datasets/indo-mmarco-500k.json", orient="records", lines=True, force_ascii=True
)
