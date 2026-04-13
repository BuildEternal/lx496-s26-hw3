# Written Problems

## Problem 1

### Problem 1b

`'input_ids'` is most likely the actual token embeddings of the given sentences. `'token_type_ids'`, based on one of the
diagrams in the paper, may be "segment embeddings" that determine which segment (sentence) the tokens belong to.
`'attention_mask'` may be what tells the model which tokens are actually part of the sentence rather than padding
tokens, since there is a $1$ whereever a word token is present and $0$ whenever a padding token is present.

### Problem 1c

According to the GitHub README, hyperparameter tuning was done by choosing the best hyperparameters (between batch sizes
of 8, 16, 32, 64, or 128, and learning rages of 3e-4, 1e-4, 5e-5, or 3e-5), and then training for 4 epochs.
