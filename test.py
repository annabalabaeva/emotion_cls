import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from datasets import load_dataset
import json
from time import time

from dataset_tweets import TweetsDataset, collate_texts
from rnn_model import LstmClassifier
from metrics import eval_metrics, print_conf_matrix


def test(config):
    device = torch.device(config["device"])
    data = load_dataset("dair-ai/emotion")
    dataset_test = TweetsDataset(data["test"], vocab_path=config["vocab"],
                                 load_from_file=True)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=lambda x: collate_texts(x, dataset_test.PAD_IDX)
    )

    model = LstmClassifier(len(dataset_test.tokens_vocab), config["model"])
    model.eval()
    model.load_state_dict(torch.load(config["save"]))
    model.to(device)

    use_half = False
    if config["half"]:
        print("Half precision mode.")
        model.half()
        use_half = True
        if not os.path.exists(config["half"]):
            torch.save(model.state_dict(), config["half"])

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        y_pred_full_val = []
        y_gt_full_val = []
        losses_val = []
        total_time = 0.0
        for i, batch in enumerate(dataloader_test):
            tokens = batch["tokens"].to(device)
            tokens_len = batch["tokens_len"]
            start_time = time()
            pred_prob = model(tokens, tokens_len)
            total_time += time() - start_time
            y_pred = torch.argmax(pred_prob, dim=-1)
            y_gt = batch["labels"].to(device)

            loss = criterion(pred_prob, y_gt)

            y_pred_full_val.extend(y_pred.tolist())
            y_gt_full_val.extend(y_gt.tolist())
            losses_val.append(loss.item())

        print("\n> Validation metrics")
        print(f"Loss: {np.mean(losses_val):.4f}\n")
        metrics_val = eval_metrics(
            y_pred_full_val, y_gt_full_val, print_metrics=True, by_class=True)
        _ = print_conf_matrix(y_gt_full_val, y_pred_full_val)
        print(f"\nTime per batch: {total_time / len(dataloader_test)}")


def main(config_path="config.json"):
    with open(config_path) as f:
        config = json.load(f)

    test(config)


if __name__ == "__main__":
    main()

