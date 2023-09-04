from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class HF_NLI_reader(Dataset):
    def __init__(
        self,
        tokenizer: str,
        max_token: int = 512,
        **kwargs,
    ) -> None:
        data_intermediate = load_dataset(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.data = {}
        self.max_token = max_token
        for index, split in enumerate(data_intermediate.column_names):
            self.data[index] = data_intermediate[split]

        del data_intermediate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence_1 = self.data[0][idx]
        sentence_2 = self.data[1][idx]
        print(sentence_1)
        inputs_1 = self.tokenizer(
            sentence_1, max_length=self.max_token, padding="max_length"
        )
        inputs_2 = self.tokenizer(
            sentence_2, max_length=self.max_token, padding="max_length"
        )
        return inputs_1, inputs_2
