# convert huggingface datasets to json files
from datasets import load_dataset


def dataset_to_json(
    hf_dataset: str,
    json_path: str,
):
    """
    hf_dataset: str, name of the dataset to be converted
    json_path: str, path to save the json file
    """
    dataset = load_dataset(hf_dataset)
    dataset.save_to_disk
