import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tabulate import tabulate
from dataset_tweets import CLASS_NAMES


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


def print_conf_matrix(y_gt, y_pred):
    matrix = confusion_matrix(y_gt, y_pred, normalize=None)
    matrix_with_class = [[n] + list(scores) for n, scores in zip(CLASS_NAMES, matrix)]
    print("\nConfusion matrix:")
    print(tabulate(
        matrix_with_class, headers=["class"] + CLASS_NAMES, tablefmt="github"))


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