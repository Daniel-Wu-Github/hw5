# HW5 — Question 4 Tasks (Coding Task)

This file summarizes the required tasks for Question 4 (Coding Task) and the submission checklist. Use this as your working plan and record the console logs and screenshots needed for the PDF report.

## Overview of Files

- `model.py`: Implement the core Transformer logic (Causal Self-Attention block `CSABlock`).
- `generate.py`: Implement the text generation logic (`generate_sample`).
- `main.py`: Inspect for configuration and run experiments (max length, CLI args).
- Console/Terminal: Save training/testing output logs for the PDF appendix.

## Detailed Breakdown

### Part 4(a): Tokenizer & Data Analysis
- Goal: Run the code to download data and inspect how text is tokenized.
- Files to inspect: `dataset.py`, `main.py`.
- Code to write: None.
- Report data required:
  - Tokenizer Definition: Explain what the tokenizer is and how it works.
  - Process: Describe how this tokenizer processes raw data (splitting, special tokens).
  - Vocabulary Size: Exact integer (capture from console logs or by querying the tokenizer).

### Part 4(b): Input Sequence Length
- Goal: Determine the maximum input size allowed by the model.
- Inspect `main.py` (look for parser args like `--max_len`).
- Report data required: `Max Length` integer and explanation of how it's set.

### Part 4(c): Implement Causal Self-Attention (CSABlock)
- Goal: Implement `CSABlock` in `model.py` with Q/K/V projections, scaled dot-product attention, causal mask, and value-weighting.
- Report data required: Explanation of self-attention steps, causality/mask logic, and screenshots of training/validation loss curves.

### Part 4(d): Implement Generation
- Goal: Implement autoregressive `generate_sample` in `generate.py`.
- Report data required: Explain generation process and provide an example Input → Output from the dataset.

### Part 4(e): Hyperparameter Tuning
- Goal: Run `main.py` with different `--n_layer`, `--n_head`, `--n_embd` values.
- Report data required: Table comparing runs (Layers, Heads, Embeddings, Validation Loss, Time/epoch, Test Accuracy) and short analysis.

### Part 4(f): Data Split Experiment
- Goal: Run `main.py --data_split <split_name>` for a harder split.
- Report data required: Which split, purpose, and comparison to the simple split.

## Submission Checklist

1. Code Zip: `HW5_FirstName_LastName.zip` containing modified `model.py`, `generate.py`, and any other modified files.
2. PDF Report: `HW#_FirstName_LastName.pdf` containing all answers, explanations, screenshots, and console logs.
3. Appendix: Paste the full training/testing console logs at the end of the PDF.

## Quick Notes
- Keep a copy of the complete terminal output for each experiment (use `tee` to save logs).
- Take screenshots of loss curves and key console outputs (vocabulary size, tokenizer summary).

---

## Exact Steps & Commands for Part 4(a): Tokenizer & Data Analysis

Follow these exact steps to find and record the tokenizer behavior and vocabulary size.

1) Inspect the code to locate tokenizer logic (open `dataset.py` and `main.py`):

```bash
# show the top of dataset.py and main.py
sed -n '1,240p' dataset.py | sed -n '1,240p'
sed -n '1,240p' main.py | sed -n '1,240p'
```

2) Search for likely tokenizer-related symbols (quick grep to find occurrences):

```bash
grep -nE "Tokenizer|tokenizer|vocab|vocabulary|encode|decode" dataset.py main.py || true
```

3) See available CLI options and check for `--max_len` and other flags (optional but useful):

```bash
python3 main.py --help
```

4) Run the default data-download / setup run and save logs (captures stdout/stderr):

```bash
# Run and save logs for appendix (unbuffered -u recommended)
python3 -u main.py 2>&1 | tee run_log.txt
```

5) Extract tokenizer/vocabulary information from the saved log:

```bash
# search the saved run log for vocabulary/tokenizer lines
grep -Ei "vocab|vocabulary|vocab_size|tokenizer|vocab size" run_log.txt || true
tail -n 200 run_log.txt
```

6) If the run does not print vocabulary size, run a small introspection snippet to try to instantiate/query the tokenizer defined in `dataset.py`.

```bash
python3 - <<'PY'
import importlib, inspect
mod = importlib.import_module('dataset')
cands = [(n,o) for n,o in vars(mod).items() if 'token' in n.lower() or 'Tokenizer' in n]
print('Candidates in dataset module:', [n for n,_ in cands])
tok = None
for name,obj in cands:
    try:
        if inspect.isclass(obj):
            tok = obj()
            break
        elif inspect.isfunction(obj):
            tok = obj()
            break
    except Exception:
        continue
if tok is None:
    print('No tokenizer instance found automatically; open dataset.py and inspect how tokenizer is built.')
else:
    print('Tokenizer instance type:', type(tok))
    for attr in ('vocab_size','n_vocab','vocab','vocab_size_','vocabulary'):
        if hasattr(tok,attr):
            print(attr, getattr(tok,attr))
    try:
        enc = tok.encode('this is a test')
        print('encode("this is a test") ->', enc)
        print('decode back ->', tok.decode(enc))
    except Exception as e:
        print('encode/decode failed:', e)
PY
```

7) Record the required report items:

- **Tokenizer Definition:** Write a short paragraph describing what a tokenizer is (e.g., mapping text to integer tokens, handling special tokens like `<eos>`).
- **Process:** Describe, from reading `dataset.py`, how raw text is tokenized (split rules, special tokens, lowercasing, etc.). Quote the exact code lines you used as evidence.
- **Vocabulary Size:** Copy the exact integer found in `run_log.txt` or from the introspection snippet output. Paste that integer into your report.

8) Save screenshots and the log file `run_log.txt` for the PDF appendix.

Good luck — after you complete these steps, come back with the `run_log.txt` or the printed vocabulary size and I can help you write the short paragraphs and format the PDF-ready screenshots and appendices.
