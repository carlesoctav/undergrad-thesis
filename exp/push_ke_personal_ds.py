from dotenv import load_dotenv
import os
import datasets

if __name__ == "__main__":
    load_dotenv()
    dataset = datasets.load_dataset("genta-tech/snli_indo")

    dataset.push_to_hub(
        "carles-undergrad-thesis/indol-snli", token=os.getenv("HF_WRITE")
    )
