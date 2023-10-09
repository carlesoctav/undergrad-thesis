import pytorch_lightning as pl
import os
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from modules.models.embedder import SentenceEmbedder
from modules.trainer.TripletTrainer import TripletTrainer
from modules.dataset.mmarco_dataset import MmarcoDataset
from transformers import AutoTokenizer
import yaml
import argparse
from torch.utils.data import DataLoader
from dotenv import load_dotenv


load_dotenv()


def batch_tokenize(batch):
    global tokenizer

    sentence_1 = []
    sentence_2 = []
    sentence_3 = []

    for item in batch:
        sentence_1.append(item[0])
        sentence_2.append(item[1])
        sentence_3.append(item[2])

    sentence_1 = tokenizer(
        sentence_1,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=args["model"]["max_length"],
    )
    sentence_2 = tokenizer(
        sentence_2,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=args["model"]["max_length"],
    )

    sentence_3 = tokenizer(
        sentence_3,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=args["model"]["max_length"],
    )

    return sentence_1, sentence_2, sentence_3


def train_model_triplet_mmarco(**kwargs):
    print(kwargs)

    train_model = None
    device = None
    test_loader = None

    if kwargs["training"]["accelerator"] == "gpu":
        device = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    print(f"==>> device: {device}")

    embedder = SentenceEmbedder(
        pretrained_model=kwargs["model"]["pretrained_model"],
        pooling_layer=kwargs["model"]["pooling_layer"],
        normalize_layer=kwargs["model"]["normalize_layer"],
    )

    finetuned_file = os.path.join(
        kwargs["training"]["default_root_dir"] + "/best_model.ckpt"
    )

    trainer = pl.Trainer(
        default_root_dir=kwargs["training"]["default_root_dir"],
        accelerator=device,
        max_epochs=kwargs["training"]["max_epochs"],
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                dirpath=kwargs["training"]["default_root_dir"],
                filename="best_model",
            ),
        ],
        gradient_clip_val=5.0,
        logger=[
            pl.loggers.TensorBoardLogger(
                save_dir=kwargs["training"]["default_root_dir"],
                name="logs",
                version="0",
            ),
            pl.loggers.CSVLogger(
                save_dir=kwargs["training"]["default_root_dir"],
                name="logs",
                version="0",
            ),
        ],
    )

    train_ds = MmarcoDataset(kwargs["dataset"]["train"])
    train_loader = DataLoader(
        train_ds,
        batch_size=kwargs["dataset"]["batch_size"],
        shuffle=True,
        num_workers=kwargs["dataset"]["num_workers"],
        collate_fn=batch_tokenize,
    )

    val_ds = MmarcoDataset(kwargs["dataset"]["val"])
    val_loader = DataLoader(
        val_ds,
        batch_size=kwargs["dataset"]["batch_size"],
        shuffle=False,
        num_workers=kwargs["dataset"]["num_workers"],
        collate_fn=batch_tokenize,
    )

    if kwargs["dataset"]["test"] != None:
        test_ds = MmarcoDataset(kwargs["dataset"]["test"])
        test_loader = DataLoader(
            test_ds,
            batch_size=kwargs["dataset"]["batch_size"],
            shuffle=False,
            num_workers=kwargs["dataset"]["num_workers"],
            collate_fn=batch_tokenize,
        )

    if os.path.isfile(finetuned_file):
        print(
            f"Found pretrained model at {finetuned_file}, loading... unfortunately not supported yet"
        )
    else:
        pl.seed_everything(kwargs["additional"]["seed"])
        train_model = TripletTrainer(
            embedder=embedder,
            max_iters=kwargs["training"]["max_epochs"] * len(train_loader),
            triplet_margin=kwargs["model"]["triplet_margin"],
            distance_metric=kwargs["model"]["distance_metric"],
            **kwargs["optimizer"],
        )
        if kwargs["additional"]["train"]:
            trainer.fit(
                model=train_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

    try:
        test_result = trainer.test(train_model, test_loader, verbose=True)
    except Exception as e:
        print(f"==>> Exception: {e}")

    if kwargs["additional"]["push_to_hub"]:
        print(f"Pushing to hub...")
        embedder.model.push_to_hub(
            kwargs["additional"]["model_name_hub"], use_auth_token=os.getenv("HF_TOKEN")
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/nli_model/config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    args = parse_args()
    print(f"==>> args: {args}")
    tokenizer = AutoTokenizer.from_pretrained(args["model"]["pretrained_model"])
    train_model_triplet_mmarco(**args)
