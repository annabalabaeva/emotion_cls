import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from datasets import load_dataset
from tabulate import tabulate
from tqdm import tqdm

from test import eval_metrics

BATCH_SIZE = 8


def run_test(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    emotion_analysis = \
        pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    y_pred_list = np.zeros(len(dataset), np.int32)
    y_true_list = np.zeros(len(dataset), np.int32)
    for i in tqdm(np.arange(0, len(dataset), BATCH_SIZE)):
        y_pred_dict = emotion_analysis(dataset["text"][i: i + BATCH_SIZE])
        y_pred = [d["label"] for d in y_pred_dict]
        y_true = dataset["label"][i: i + BATCH_SIZE]
        y_pred_list[i: i + BATCH_SIZE] = [int(name[-1]) for name in y_pred]
        y_true_list[i: i + BATCH_SIZE] = y_true

    return accuracy(y_true_list, y_pred_list), \
        prec_recall_f1(y_true_list, y_pred_list, by_class=True)


def main():
    print("Loading dataset...")
    dataset = load_dataset("dair-ai/emotion")["test"]

    print("\nTesting models...\n")
    model_name = "Vasanth/bert-base-uncased-finetuned-emotion"
    acc, stat = run_test(model_name, dataset)

    print(f"Accuracy: {acc:4f}")
    stat = [[n] + list(s) for n, s in zip(CLASS_NAMES, stat)]
    print(tabulate(
        stat, headers=["class", "Precision", "Recall", "F1-score", "Support"],
        tablefmt="github", floatfmt=("s", ".3f", ".3f", ".3f", ".0f")))


if __name__ == "__main__":
    main()
