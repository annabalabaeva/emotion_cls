import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from datasets import load_dataset
from tqdm import tqdm
from time import time

from test import eval_metrics

BATCH_SIZE = 32


def run_test(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    emotion_analysis = \
        pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    y_pred_list = np.zeros(len(dataset), np.int32)
    y_true_list = np.zeros(len(dataset), np.int32)
    total_time = 0.0
    for i in tqdm(np.arange(0, len(dataset), BATCH_SIZE)):
        start_time = time()
        y_pred_dict = emotion_analysis(dataset["text"][i: i + BATCH_SIZE])
        total_time += time() - start_time
        y_pred = [d["label"] for d in y_pred_dict]
        y_true = dataset["label"][i: i + BATCH_SIZE]
        y_pred_list[i: i + BATCH_SIZE] = [int(name[-1]) for name in y_pred]
        y_true_list[i: i + BATCH_SIZE] = y_true

    n_batches = np.ceil(len(dataset) / BATCH_SIZE)

    print(f"Time per batch: {total_time / n_batches}")

    return y_true_list, y_pred_list


def main():
    print("Loading dataset...")
    dataset = load_dataset("dair-ai/emotion")["test"]

    print("\nTesting models...\n")
    model_name = "Vasanth/bert-base-uncased-finetuned-emotion"
    y_true, y_pred = run_test(model_name, dataset)

    metrics_val = eval_metrics(y_pred, y_true, print_metrics=True)


if __name__ == "__main__":
    main()
