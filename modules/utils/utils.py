# convert huggingface datasets to json files
from datasets import load_dataset
import os


def convert_hf_nli_dataset_to_json(dataset_name, split_name):
    dataset = load_dataset(dataset_name, split=split_name)
    print(f"==>> dataset: {dataset}")

    if not os.path.exists("data"):
        os.makedirs("data")

    dataset = dataset.rename_column("premise", "sentence_1")
    dataset = dataset.rename_column("hyphothesis", "sentence_2")

    dataset.to_json(f"data/{dataset_name}_{split_name}.json", orient="records")


if __name__ == "__main__":
    convert_hf_nli_dataset_to_json("carles-undergrad-thesis/indo-snli", "train")
    convert_hf_nli_dataset_to_json("carles-undergrad-thesis/indo-snli", "validation")
    convert_hf_nli_dataset_to_json("carles-undergrad-thesis/indo-snli", "test")
