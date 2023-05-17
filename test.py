import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tabulate import tabulate
from dataset_tweets import TweetsDataset, CLASS_NAMES, read_vocab

from torch.utils.data import DataLoader
from datasets import load_dataset
import json

from dataset_tweets import TweetsDataset, collate_texts
from rnn_model import LstmClassifier


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

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        y_pred_full_val = []
        y_gt_full_val = []
        losses_val = []
        for i, batch in enumerate(dataloader_test):
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


def accuracy(y_gt, y_pred):
    return accuracy_score(y_gt, y_pred)


def prec_recall_f1(y_gt, y_pred, by_class=False):
    if by_class:
        stat = precision_recall_fscore_support(
            y_gt, y_pred, average=None, zero_division=0)
        stat = np.transpose(np.array(stat))
        return stat

    else:
        return precision_recall_fscore_support(
            y_gt, y_pred, average='macro', zero_division=0)


def eval_metrics(y_pred, y_gt, by_class=False, print_metrics=False):
    acc = accuracy(y_gt, y_pred)
    stat = prec_recall_f1(y_gt, y_pred, by_class=by_class)

    if by_class:

        if print_metrics:
            stat = [[n] + list(s) for n, s in zip(CLASS_NAMES, stat)]
            print(tabulate(
                stat, headers=["class", "Precision", "Recall", "F1-score", "Support"],
                tablefmt="github", floatfmt=("s", ".3f", ".3f", ".3f", ".0f")))
        return acc, stat

    else:
        prec, recall, f1_score, _ = stat
        if print_metrics:
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {prec:.4f}, "
                  f"Recall: {recall:.4f}, "
                  f"F1-score: {f1_score:.4f}")
        return {
            "acc": acc,
            "prec": prec,
            "recall": recall,
            "f1-score": f1_score
        }


def main(config_path="config.json"):
    with open(config_path) as f:
        config = json.load(f)

    test(config)

if __name__ == "__main__":
    main()

