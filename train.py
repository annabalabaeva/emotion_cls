import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import json

from dataset_tweets import TweetsDataset, collate_texts
from rnn_model import LstmClassifier
from metrics import eval_metrics


def train(config, model, dataloaders):
    print("Start training.")
    device = torch.device(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    model.to(device)

    y_pred_full = []
    y_gt_full = []
    losses = []
    best_f1 = 0.0
    best_epoch = -1
    for epoch_i in range(config["n_epochs"]):
        print(f"\nEpoch: {epoch_i}")
        # Training
        model.train()
        for i, batch in enumerate(dataloaders["train"]):
            tokens = batch["tokens"].to(device)
            tokens_len = batch["tokens_len"]
            pred_prob = model(tokens, tokens_len)
            y_pred = torch.argmax(pred_prob, dim=-1)
            y_gt = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = criterion(pred_prob, y_gt)
            loss.backward()
            optimizer.step()

            y_pred_full.extend(y_pred.tolist())
            y_gt_full.extend(y_gt.tolist())
            losses.append(loss.item())

        print("\n> Train metrics")
        print(f"Loss: {np.mean(losses):.4f}\n")
        metrics_train = eval_metrics(y_pred_full, y_gt_full, print_metrics=True)

        # Validation
        model.eval()
        with torch.no_grad():
            y_pred_full_val = []
            y_gt_full_val = []
            losses_val = []
            for i, batch in enumerate(dataloaders["val"]):
                tokens = batch["tokens"].to(device)
                tokens_len = batch["tokens_len"]
                pred_prob = model(tokens, tokens_len)
                y_pred = torch.argmax(pred_prob, dim=-1)
                y_gt = batch["labels"].to(device)

                loss = criterion(pred_prob, y_gt)

                y_pred_full_val.extend(y_pred.tolist())
                y_gt_full_val.extend(y_gt.tolist())
                losses_val.append(loss.item())

            print("\n> Validation metrics")
            print(f"Loss: {np.mean(losses_val):.4f}\n")
            metrics_val = eval_metrics(
                y_pred_full_val, y_gt_full_val, print_metrics=True)

        f1_val = metrics_val["f1-score"]
        if f1_val > best_f1:
            best_f1 = f1_val
            best_epoch = epoch_i
            torch.save(model.cpu().state_dict(), config["save"])
            model.to(device)
            print("\nModel was saved.")

    print(f"\n------\nBest model was saved on {best_epoch} epoch");


def main(config_path="config.json"):
    with open(config_path) as f:
        config = json.load(f)

    datasets = init_datasets(config)
    dataloaders = init_dataloaders(config, datasets)

    vocab_size = len(datasets["train"].tokens_vocab)
    model_cfg = config["model"]
    model = LstmClassifier(vocab_size, model_cfg)

    train(config, model, dataloaders)


def init_datasets(config):
    print("Loading data...")
    data = load_dataset("dair-ai/emotion")
    print("Building dataset...")
    dataset_train = TweetsDataset(data["train"], vocab_path=config["vocab"])
    dataset_val = TweetsDataset(data["validation"], dataset_train.tokens_vocab)
    dataset_test = TweetsDataset(data["test"], dataset_train.tokens_vocab)

    return {
        "train": dataset_train,
        "val": dataset_val,
        "test": dataset_test
    }


def init_dataloaders(config, datasets):
    dataloaders = dict()
    dataloaders["train"] = DataLoader(
        datasets["train"],
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=True,
        collate_fn=lambda x: collate_texts(x, datasets["train"].PAD_IDX)
    )

    dataloaders["val"] = DataLoader(
        datasets["val"],
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=lambda x: collate_texts(x, datasets["train"].PAD_IDX)
    )

    dataloaders["test"] = DataLoader(
        datasets["test"],
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=lambda x: collate_texts(x, datasets["train"].PAD_IDX)
    )
    return dataloaders


if __name__ == "__main__":
    main()
