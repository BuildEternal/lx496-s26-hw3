"""
Code for Problem 1 of HW 2.
"""

from functools import partial
import pickle
from typing import Any, Dict

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction  # type: ignore


def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer
    """

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, return_tensors="np")

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> BertForSequenceClassification:
    """
    Problem 2a: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    model = BertForSequenceClassification.from_pretrained(model_name)
    if use_bitfit:
        for name, param in model.named_parameters():
            if not name.endswith(".bias"):
                param.requires_grad = False
    return model


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    preds = np.argmax(p.predictions, axis=-1)
    acc = (preds == p.label_ids).mean()
    return {"accuracy": float(acc)}


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset, use_bitfit: bool = False) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=2e-5,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )
    return Trainer(
        model_init=partial(init_model, model_name=model_name, use_bitfit=use_bitfit),
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 2c: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    return {
        "direction": "maximize",
        "n_trials": 20,
        "backend": "optuna",
        "hp_space": lambda trial: {
            "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 1e-4, 5e-5, 3e-5]),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8, 16, 32, 64, 128]
            ),
        },
        "compute_objective": lambda metrics: metrics["eval_accuracy"],
    }


if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(0.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"], use_bitfit=True)

    # Train and save the best hyperparameters
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results.p", "wb") as f:
        pickle.dump(best, f)
