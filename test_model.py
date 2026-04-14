"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import compute_metrics, preprocess_dataset


def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    model = BertForSequenceClassification.from_pretrained(directory)

    args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_eval_batch_size=8,
    )

    return Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
    )


if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester("path_to_your_best_model")

    # Test
    results = tester.predict(imdb["test"]) # type: ignore
    with open("test_results.p", "wb") as f:
        pickle.dump(results, f)
