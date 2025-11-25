CSCE 421 Assignment #5 - Command Log & Explanations

This document tracks the terminal commands used to complete the coding assignment.

1. Environment Setup (Critical Fix)

Issue: The datasets library version 3.0+ removed support for the custom scan.py script used by this dataset, causing a RuntimeError.
Command:

pip install "datasets<3.0.0"


Explanation: We downgraded the library to a 2.x version (likely 2.21.0) which still supports trust_remote_code=True and custom loading scripts.

2. Part 4(a): Tokenizer Analysis

Goal: Download the data and find the vocabulary size.
Command:

python main.py --task train --run_name hw5_test


Explanation: We ran the training task before implementing the model. This triggered the data download and tokenizer construction. It printed the vocabulary (size: 23) to the console before crashing (as expected) because model.py was empty at the time.

3. Part 4(c): Training the Baseline Model

Goal: Verify CSABlock implementation and get baseline results.
Command:

python main.py --task train --run_name hw5_test


Explanation: After implementing the CSABlock in model.py, we ran the default training loop.

Hyperparameters: Default (n_layer=2, n_head=2, n_embd=16).

Result: Loss decreased from ~1.35 to ~0.35, proving the Attention Mechanism works, but the model capacity was too low for high performance.

4. Part 4(d): Testing Generation (Baseline)

Goal: Verify generate_sample implementation.
Command:

python main.py --task generate --run_name hw5_test


Explanation: We ran the generation script using the baseline model weights.

Result: Accuracy was ~7.6%. This confirmed the generation logic (Greedy Decoding) was functional, even though the model itself was "dumb" due to the small embedding size.

5. Part 4(e): Hyperparameter Tuning (The Fix)

Goal: Achieve high accuracy (>90%) by increasing model capacity.

Step 1: Training

Command:

python main.py --task train --run_name hw5_tuned --n_embd 128 --n_layer 4 --n_head 8


Explanation: We increased the model size significantly:

Embeddings: 16 $\to$ 128

Layers: 2 $\to$ 4

Heads: 2 $\to$ 8

Result: Validation loss dropped to near zero (~0.0001).

Step 2: Testing

Command:

python main.py --task generate --run_name hw5_tuned --n_embd 128 --n_layer 4 --n_head 8


Explanation: We tested the tuned model. Result: ~99.8% Accuracy.

6. Part 4(f): Data Splits (Generalization)

Goal: Test if the model can generalize to new sequence lengths.

Step 1: Training on length split

Command:

python main.py --task train --run_name hw5_split_test --data_split length --n_embd 128 --n_layer 4 --n_head 8


Explanation: This trains the model on short commands only.

Step 2: Testing on length split

Command:

python main.py --task generate --run_name hw5_split_test --data_split length --n_embd 128 --n_layer 4 --n_head 8


Explanation: This tests the model on long commands (longer than it ever saw during training). We expect low accuracy here, demonstrating that standard Transformers struggle with systematic generalization (length extrapolation).