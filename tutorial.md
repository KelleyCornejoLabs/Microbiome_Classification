# StrataBioNN Microbiome Classification — Tutorial

*Tool by Alexander E. Symons & Omar E. Cornejo, KelleyCornejoLabs*

This is a hands-on guide to [`KelleyCornejoLabs/Microbiome_Classification`](https://github.com/KelleyCornejoLabs/Microbiome_Classification) (nicknamed **StrataBioNN**), a set of Python tools for classifying microbiome samples (e.g. vaginal community state types, oral or gut enterotypes) from bacterial count data. The centerpiece is a configurable neural network classifier, which outperforms a random forest baseline, and a collection of utilities for data splitting, evaluation, and cross-study label transfer.

Everything here was verified by running the scripts in this repository end to end, so every command below is copy-pasteable.

At a high level, working with StrataBioNN looks like:

1. **Preprocessing** the data, if it isn't already in the expected column layout
2. **Splitting** it into training / testing / validation subsets
3. **Training** a model on the training and testing subsets
4. **Evaluating** the model's performance on the held-out validation subset
5. **Classifying** new data — either data with identical columns, or (with extra steps) data from a different study with only partially overlapping columns

## Table of contents

1. [What's in the repo](#whats-in-the-repo)
2. [Installation](#installation)
3. [The data format](#the-data-format)
4. [Quickstart: split, train, and evaluate a classifier](#quickstart-split-train-and-evaluate-a-classifier)
5. [Classifying new, unlabeled samples](#classifying-new-unlabeled-samples)
6. [Applying a classification scheme to a new study](#applying-a-classification-scheme-to-a-new-study)
7. [Random forest baseline](#random-forest-baseline)
8. [Comparing classifiers head-to-head](#comparing-classifiers-head-to-head)
9. [Utility script reference](#utility-script-reference)
10. [Full CLI reference for nn_classifier.py](#full-cli-reference-for-nn_classifierpy)
11. [Tips and troubleshooting](#tips-and-troubleshooting)

## What's in the repo

```text
Microbiome_Classification/
├── data/                      # Example datasets (vaginal, oral, gut) already in VALENCIA-style CSV format
├── scripts/
│   ├── nn_classifier.py       # Main neural network classifier (train / classify / evaluate / inspect)
│   ├── random_forest_classifier.py   # Random forest baseline (imports nn_classifier for data loading)
│   ├── find_VAL_overlap.py    # Finds bacteria species shared between two datasets
│   ├── environment.yaml       # Conda environment definition
│   └── utilities/
│       ├── make_test_train_split.py  # Splits a labeled CSV into train/test/validation sets
│       ├── preprocess_valencia.py    # Reformats VALENCIA-repo data for use with VALENCIA itself
│       ├── eval_valencia.py          # Accuracy + confusion matrix for any classifier's predictions
│       ├── evaluate_valencia.sh      # End-to-end VALENCIA evaluation
│       ├── test_multiple.sh          # Compares VALENCIA vs. neural net vs. random forest
│       ├── centroids.py               # Computes class centroids (e.g. for VALENCIA)
│       ├── check_tolerances.py        # Sanity-checks class balance across a data split
│       ├── pacmap_graph.py            # 2D PaCMAP projection of samples, colored by label
│       ├── oral_preprocessor.py       # K-means-based labeling scheme for oral microbiome data
│       ├── nn_trainer.py              # Hyperparameter sweep over the neural classifier
│       └── process_oral.sh            # Runs the oral preprocessor then splits the output
└── README.md
```

All scripts here take `.csv` files in the same column layout as the [VALENCIA](https://github.com/ravel-lab/VALENCIA) project, so once your data is in that shape, every tool in the repo can read it.

## Installation

Clone the repository:

```bash
git clone https://github.com/KelleyCornejoLabs/Microbiome_Classification.git
cd Microbiome_Classification
```

**Option A — conda (recommended):**

```bash
conda env create -f scripts/environment.yaml
conda activate stratabionn
```

This installs Python 3.13, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, and `torch` (via pip inside the conda env).

**Option B — plain pip:**

```bash
pip install pandas numpy matplotlib scikit-learn torch
```

**Optional extras** used by a couple of the utility scripts, install only if you need them:

```bash
pip install conorm   # enables TMM normalization (--normalizing-function tmm) in nn_classifier.py
pip install pacmap    # required by scripts/utilities/pacmap_graph.py
```

If `conorm` isn't installed, `nn_classifier.py` will just print a warning and disable the `tmm` normalization option — everything else still works.

Every command in this tutorial is run **from the repository root**, so paths like `scripts/nn_classifier.py` and `data/vaginal/France/...` resolve correctly.

## The data format

All input CSVs are one row per sample, one column per bacterial taxon (raw counts or relative abundance), plus a few reserved columns:

| Column        | Required for                                   | Notes                                                            |
|---------------|-------------------------------------------------|-------------------------------------------------------------------|
| `sampleID`    | everything                                       | Unique identifier per sample                                      |
| `read_count`  | everything                                       | Total reads for that sample                                       |
| `HC_subCST`   | training data, and labeled test/validation data  | The class label to predict. Unlabeled data simply omits this column |

These names come from VALENCIA's own conventions. If your dataset uses different column names, `make_test_train_split.py` (below) can rename them for you as part of the split.

The repo already ships a real example dataset at `data/vaginal/France/all_samples_taxonomic_composition_data.csv` — the raw vaginal microbiome dataset from the VALENCIA project, ~13,200 samples labeled with community state types (CSTs like `I-A`, `III-B`, `IV-C2`, etc.). We'll use it throughout this tutorial.

## Quickstart: split, train, and evaluate a classifier

### 1. Split the data into train / test / validation

`make_test_train_split.py` performs a **stratified** split, meaning each class (each `HC_subCST` label) is split in the same proportions, so rare classes aren't accidentally left out of one of the sets.

```bash
python3 scripts/utilities/make_test_train_split.py \
  -i data/vaginal/France/all_samples_taxonomic_composition_data.csv \
  -s 80 -v 10 \
  -o valencia_data \
  -t 0.0015
```

- `-s 80` — 80% of each class goes to the training set
- `-v 10` — 10% goes to a validation set (the remainder, ~10%, becomes the test set)
- `-t 0.0015` — the maximum allowed *absolute* difference between a split's class balance and the original data's class balance, both expressed as decimal fractions (0–1), not percentages. For example, if a class makes up 5% of the full dataset (`0.05`), a tolerance of `0.0015` allows that class's share in any split to land anywhere between `0.0485` and `0.0515` (i.e. within 0.15 percentage points) before the script complains. If a split can't meet this tolerance, the script prints exactly which class is out of bounds and by how much — raise `-t` if that happens, which is common with rare classes where a single sample can swing the percentage a lot.
- `-o valencia_data` — output prefix

This produces `valencia_data_train.csv`, `valencia_data_test.csv`, and `valencia_data_validation.csv`. Because we didn't pass `-sid`/`-rc`/`-lc`/`-nd`, the script falls back to defaults tailored for VALENCIA-repo data specifically: it looks for `Sample_number_for_SRA` (sample ID), `total_reads` (read count), and `HC_subCST` (label), and it strips out VALENCIA's own prediction/similarity columns (`Val_CST`, `Val_subCST`, `I-A_sim` through `V_sim`, `Subject_number`, `HC_CST`) before writing the split. Since our example file is exactly that format, it just works. Written out explicitly, the command above is equivalent to:

```bash
python3 scripts/utilities/make_test_train_split.py \
  -i data/vaginal/France/all_samples_taxonomic_composition_data.csv \
  -o valencia_data -s 80 -v 10 -t 0.0015 \
  -sid Sample_number_for_SRA -rc total_reads -lc HC_subCST \
  -nd "Val_CST,Val_subCST,I-A_sim,I-B_sim,II_sim,III-A_sim,III-B_sim,IV-A_sim,IV-B_sim,IV-C0_sim,IV-C1_sim,IV-C2_sim,IV-C3_sim,IV-C4_sim,V_sim,Subject_number,HC_CST"
```

For a dataset with different column names, override whichever of these you need:

```bash
python3 scripts/utilities/make_test_train_split.py \
  -i my_other_dataset.csv \
  -sid Sample_ID -rc Reads -lc CST_label \
  -nd "notes,batch" \
  -s 80 -v 10 -o my_dataset
```

- `-sid` — column holding sample IDs (renamed to `sampleID`)
- `-rc` — column holding read counts (renamed to `read_count`)
- `-lc` — column holding class labels (renamed to `HC_subCST`)
- `-nd` — comma-separated list of any other non-count columns to strip out (metadata, notes, etc.)
- `-tr` — pass this flag if your samples are laid out as columns instead of rows

### 2. Train a neural network classifier

```bash
python3 scripts/nn_classifier.py \
  -ite data/vaginal/France/formatted_80_test.csv \
  -itr data/vaginal/France/formatted_80_train.csv \
  -p saved_model
```

- `-itr` / `--input-train` — training set (patterns are learned from this)
- `-ite` / `--input-test` — held-out test set, used during training to detect overfitting and pick the best-performing model
- `-p` / `--path` — prefix for the saved files: `saved_model_nn.pt` (the trained network), `saved_model_metrics.txt` (a training log), and `saved_model_plt.png` (a loss-curve plot) — all three are written automatically, regardless of `--debug`

Training uses learning-rate scheduling: it trains until the learning rate decays below `--threshhold-lr` (default `0.00001`) or `--max-epochs` is hit (default `50000`), whichever comes first — so a run can finish in seconds on a small model or take a while on a larger one.

Add `--debug` if you also want live per-epoch progress printed to the console (loss, test loss, accuracy, and learning rate at every `--metrics-interval` epochs) while training runs. It's off by default, so training is quiet unless you ask for it — pass `--no-debug` explicitly if you want to be certain it's off. Note that `--debug` only affects console verbosity during training; it does **not** control whether the metrics file or loss-curve PNG get written (they always do), and — as you'll see in the next step — it doesn't gate the plots that `--test-accuracy` produces either.

Real output from a short training run on the command above (200 epochs, for illustration — your numbers will differ since the network's starting weights are randomized and a full run trains much longer):

```text
saved_model_metrics.txt:
Final Max Accuracy: 94.38% in 0:0:10
Test Config: lr: 0.9, linear: False, loss_fn: ce, optim: sgd
Epoch: 0,   Train loss: 2.605, Test Loss: 2.510, Accuracy: 21.72
Epoch: 50,  Train loss: 0.275, Test Loss: 0.228, Accuracy: 91.88
Epoch: 100, Train loss: 0.199, Test Loss: 0.167, Accuracy: 93.85
Epoch: 150, Train loss: 0.172, Test Loss: 0.149, Accuracy: 94.38
```

### 3. Evaluate the trained model on the validation set

The test set was already "seen" indirectly during training (it's what early stopping is based on), so use the validation set — which the model has never touched — to get an honest accuracy estimate:

```bash
python3 scripts/nn_classifier.py \
  -ite data/vaginal/France/formatted_80_validation.csv \
  -p saved_model \
  --test-accuracy
```

`--test-accuracy` does more than just report a number. In one run it will:

1. Print overall accuracy and a weighted F1 score
2. Print a text confusion matrix — **and** pop up a graphical confusion-matrix window and an interactive 2D scatterplot (with buttons to page through each pair of bacterial features and a text box to bring a chosen class to the front) — both plots appear unconditionally, regardless of whether `--debug`/`--no-debug` is set
3. Print a **cumulative guess ranking**: how many samples were correct on the model's 1st choice, how many more would have been correct on the 2nd choice, and so on
4. Run a **perturbation (feature-importance) analysis** and write four CSVs — `perturbation_analysis_f1.csv`, `perturbation_analysis_recall.csv`, `perturbation_analysis_precision.csv`, and `perturbation_analysis_accuracy.csv` — to the working directory

A validation run against the model above produced:

```text
accuracy: 93.78%
F1 (weighted): 0.9372
Correct guesses
1st guess, 2nd guess, ...
[1251, 76, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Running perterbation analysis...
Writing analysis for f1...
Writing analysis for recall...
Writing analysis for precision...
Writing analysis for accuracy...
```

(Yes, the console really does print "perterbation" — a small typo in the source, harmless to ignore.)

Here, 1251 of 1334 validation samples were correct on the first guess, another 76 would have been correct on the second guess, and so on — a useful diagnostic for spotting classes that are frequently confused with one specific neighbor.

Each perturbation CSV has one row per bacterial feature and one column per class, e.g.:

```text
feature,I-A,I-B,II,III-A,III-B,IV-A,IV-B,IV-C0,IV-C1,IV-C2,IV-C3,IV-C4,V
g_Acidaminococcus,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
Lactobacillus_iners,0.02,0.01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
```

Each cell is how much that class's score for that metric *drops* when that one feature's values are randomly shuffled across samples (baseline minus permuted). Larger values mean the model relies more heavily on that feature to identify that class — a good way to sanity-check that a model is keying off biologically sensible species rather than noise. Since this analysis re-evaluates the model once per feature per metric, it can take noticeably longer than the rest of `--test-accuracy` on datasets with hundreds of features.

If you're running on a headless server and don't want the plot windows to try to open, either run with a non-interactive matplotlib backend (`MPLBACKEND=Agg python3 scripts/nn_classifier.py ...`) or run in an environment without a display — matplotlib will skip the actual rendering but the script will still run through to completion.

### 4. Inspect a saved model at any time

```bash
python3 scripts/nn_classifier.py -p saved_model --info
```

Prints the network's architecture, the exact list of features (bacterial taxa) it expects as input, the classes it predicts, and the optimizer/learning rate it was trained with — handy for confirming a model matches the dataset you're about to feed it.

## Classifying new, unlabeled samples

Once you have a trained model, use `--classify` to label new data. Point `-ite` at the file to classify and `-out` at where to write the results:

```bash
python3 scripts/nn_classifier.py \
  -ite new_samples.csv \
  -p saved_model \
  --classify \
  -out new_samples_classified
```

The output is written to `new_samples_classified.csv` (`-out` doesn't need the extension — it's added automatically). It's the input data with two kinds of columns added:

- **`subCST`** — the predicted label for each sample
- **`Pct <label>`** — one column per class (e.g. `Pct I-A`, `Pct III-B`, ...), holding the model's softmax confidence that the sample belongs to that class, as a value between 0 and 1

`new_samples.csv` must contain the same feature columns the model was trained on (check with `--info` if unsure) — it does **not** need an `HC_subCST` column, since that's what you're generating.

There is also a `-lb`/`--labeled` flag for classify mode, intended for cases where your input already has true labels. As of this writing it's accepted but doesn't change the output — classify mode always just writes predictions and confidences. If you want to check accuracy on a file you already know the true labels for, classify it as above and then compare the result against the ground truth with `eval_valencia.py` (see the [utility script reference](#utility-script-reference)), or use `--test-accuracy` directly if you don't need per-sample output.

## Applying a classification scheme to a new study

A common situation: you have a labeled dataset (e.g. the VALENCIA vaginal data) and a second, unlabeled dataset from a different study that doesn't share every bacterial species name or column with the first. This section mirrors the cross-study workflow the repo's README calls out, using the VALENCIA data and a second dataset (`hickey.csv`, following the README's example — substitute your own second dataset here) as an example.

### Step 1 — Find the species both datasets have in common

```bash
python3 scripts/find_VAL_overlap.py --files data/vaginal/France/formatted_80_train.csv data/vaginal/Hickey/hickey_formatted.csv
```

This normalizes and compares column names across every file you pass in, and prints a ready-to-paste, comma-separated list under the header `For use in nn_classifier:`.

### Step 2 — Train a "simple" model restricted to those shared species

The neural classifier can be restricted to a specific feature set with `--focus-columns`. Combine this with `--no-train` (skip full training) and `--train-simple` (do train the restricted model) to build a model that only looks at the overlapping species:

```bash
python3 scripts/nn_classifier.py \
  -ite data/vaginal/France/formatted_80_test.csv -itr data/vaginal/France/formatted_80_train.csv \
  -p simple_model \
  --train-multiple 3 --no-train --train-simple \
  --focus-columns acidaminococcus,acinetobacter,actinobaculum,lactobacillus_iners,gardnerella_vaginalis,prevotella_bivia
```

(In practice, paste in the full comma-separated list produced by Step 1 — the short list above is just illustrative.)

A few notes on this command, since the flag names are a little confusing:

- `--no-train` skips the *initial* full-feature training pass; this feature was originally built to run after a full model was already trained, so `--no-train` is what lets you skip straight to the restricted model.
- `--train-multiple 3` trains 3 candidate models and keeps the best one — smaller, restricted-feature models are more sensitive to bad random initialization, so training a few and picking the winner helps.
- `--focus-columns` only affects the *simple* model here, not a full one.

### Step 3 — Label the new dataset with the simple model

```bash
python3 scripts/nn_classifier.py \
  -ite data/vaginal/Hickey/hickey_formatted.csv \
  -p simple_model \
  --classify \
  -out hickey_classified
```

### Step 4 — Turn the labeled output into a proper training set

The classifier's output includes per-class confidence percentage columns alongside the prediction — strip those out and split the result into train/test/validation with `make_test_train_split.py`'s `--non-data` flag:

```bash
python3 scripts/utilities/make_test_train_split.py \
  -i hickey_classified.csv \
  -o hickey_dataset \
  -sid sampleID -rc read_count -lc subCST \
  -nd "Pct I-A,Pct I-B,Pct II,Pct III-A,Pct III-B,Pct IV-A,Pct IV-B,Pct IV-C0,Pct IV-C1,Pct IV-C2,Pct IV-C3,Pct IV-C4,Pct V" \
  -t 0.01 -s 80
```

You now have a training set built from a completely different study, labeled according to the original dataset's classification scheme — ready to train a fresh, full-featured model on with the same steps from the [Quickstart](#quickstart-split-train-and-evaluate-a-classifier) above.

## Random forest baseline

`random_forest_classifier.py` reuses `nn_classifier.py`'s data loading, so it accepts the same CSVs and shares the same "train on one set, evaluate on another" pattern:

```bash
python3 scripts/random_forest_classifier.py \
  -itr data/vaginal/France/formatted_80_train.csv \
  -ite data/vaginal/France/formatted_80_validation.csv \
  --test-accuracy \
  -o rf_predictions.csv
```

- `--test-accuracy` prints accuracy and (unless you pass `--no-debug`) shows a confusion matrix pop-up
- `-o` writes predictions (with a `subCST` column, same convention as the neural classifier) to a CSV

The model trains 10,000 trees by default (`sqrt`-of-features max features per split), so expect this to take noticeably longer than the neural network on the same data — it's meant as an accuracy baseline, not a fast option.

## Comparing classifiers head-to-head

If you also have [VALENCIA](https://github.com/ravel-lab/VALENCIA) itself checked out (as a sibling directory, `../VALENCIA`), `scripts/utilities/test_multiple.sh` runs VALENCIA, the neural network, the neural network's simplified variant, and the random forest classifier back-to-back on the same test set, printing an accuracy and confusion matrix for each:

```bash
cd scripts/utilities
# Edit the path variables at the top of the script first — see below
./test_multiple.sh
```

Open the script and set these paths before running (they're placeholders by default):

```bash
valencia_path="../VALENCIA/Valencia.py"
test_centroids="../VALENCIA/CST_centroids_012920.csv"
train_set_path="./out_train.csv"     # your training set
test_set_path="./out_test.csv"       # your test set
```

`evaluate_valencia.sh` is the same idea but for VALENCIA alone, useful if you just want a quick accuracy number for VALENCIA on your data without running the neural or forest classifiers.

## Utility script reference

| Script | Purpose | Example |
|---|---|---|
| `scripts/utilities/preprocess_valencia.py` | Strips label/similarity columns and renames ID/read-count columns so VALENCIA-repo data can be fed to VALENCIA itself. | `python3 scripts/utilities/preprocess_valencia.py -i raw.csv -o valencia_ready.csv` |
| `scripts/utilities/eval_valencia.py` | Compares a predictions file (must have a `subCST` column) against ground truth (`HC_subCST`), prints accuracy, and saves a confusion matrix + ROC curve. | `python3 scripts/utilities/eval_valencia.py -id ground_truth.csv -ip predictions.csv -n "My Model" -o my_model_report` |
| `scripts/utilities/centroids.py` | Computes per-class mean or median feature vectors — e.g. to build a custom VALENCIA centroids file. | `python3 scripts/utilities/centroids.py data.csv -l HC_subCST -ndc sampleID,read_count -o centroids.csv` |
| `scripts/utilities/check_tolerances.py` | Reports, per class, how much a train/test/validation split's class balance deviates from the original dataset. Edit the file paths near the bottom of the script before running — it's set up to be adapted per dataset rather than driven by CLI flags. | Edit paths, then `python3 scripts/utilities/check_tolerances.py` |
| `scripts/utilities/pacmap_graph.py` | Projects samples into 2D with PaCMAP and colors them by class label — a quick visual sanity check for class separability. Requires `pip install pacmap`. | `python3 scripts/utilities/pacmap_graph.py -i valencia_data_train.csv -l HC_subCST -o pacmap_view` |
| `scripts/utilities/oral_preprocessor.py` | Reformats the Manghi et al. oral microbiome dataset and applies a K-means-based labeling scheme to produce a baseline oral dataset. | `python3 scripts/utilities/oral_preprocessor.py -in raw_oral.csv -out oral_processed.csv -cl 3` |
| `scripts/utilities/process_oral.sh` | Runs `oral_preprocessor.py` and then splits the result into train/test/validation in one step. | `./scripts/utilities/process_oral.sh` (edit the path variables at the top first) |
| `scripts/utilities/nn_trainer.py` | Sweeps combinations of optimizer, loss function, and learning rate to help find good hyperparameters. | `python3 scripts/utilities/nn_trainer.py -itr train.csv -ite test.csv -e 5000 -o sgd,adam -l ce,nll -lr 0.1,0.9` |
| `scripts/find_VAL_overlap.py` | Finds bacteria species names shared across two or more datasets, formatted for `--focus-columns`. | `python3 scripts/find_VAL_overlap.py --files a.csv b.csv` |

## Full CLI reference for nn_classifier.py

The most commonly used flags:

| Flag | Meaning |
|---|---|
| `-itr`, `--input-train` | Path to the labeled training CSV |
| `-ite`, `--input-test` | Path to the test CSV (labeled, for training/evaluation; unlabeled, for `--classify`) |
| `-p`, `--path` | Prefix used to save/load the model, metrics, and loss plot |
| `-cl`, `--classify` | Switch to classify mode: treats `--input-test` as unlabeled and produces predictions |
| `-out`, `--output` | Where to write predictions in classify mode |

All other flags (every one has a sensible default — you generally only need the five above to get started):

| Flag | Default | Meaning |
|---|---|---|
| `-tlr`, `--threshhold-lr` | `0.00001` | Stop training once the learning rate decays below this |
| `-lr`, `--learning-rate` | `0.9` | Initial learning rate |
| `-me`, `--max-epochs` | `50000` | Hard cap on training epochs |
| `-t`, `--train` | `True` | Training mode (this is the default mode, so it's rarely set explicitly) |
| `-c`, `--continue-train` | `False` | Keep training a previously saved model instead of starting fresh |
| `-m`, `--metrics-interval` | `50` | How often (in epochs) to log metrics |
| `-l`, `--loss` | `ce` | Loss function: `ce` (cross-entropy), `nll` (negative log-likelihood), or `kld` (KL divergence) |
| `-lo`, `--load` | `False` | Load an existing model as the starting point for training a simple model (used with `--path`) |
| `-o`, `--optim` | `sgd` | Optimizer: `sgd` or `adam` |
| `-li`, `--linear` | `False` | Train a linear model instead of the default non-linear (ReLU) network |
| `-pa`, `--patience` | `100` | Epochs of stagnant/negative improvement to tolerate before dropping the learning rate |
| `-sd`, `--seed` | `None` | Seed for reproducible results |
| `-hn`, `--hidden-neurons` | auto | Size of the hidden layer; default is `(2/3) × input_features + num_classes` |
| `-dbg`, `--debug` | off | Print verbose per-epoch console logging during training (use `--no-debug` to make sure it's off). Does not gate the metrics file, loss-curve PNG, or `--test-accuracy` plots — those happen either way |
| `-ts`, `--train-simple` | `False` | Also train a reduced model using only the most important features |
| `-ta`, `--test-accuracy` | `False` | Load a model, classify `--input-test`, and report accuracy/metrics |
| `-tm`, `--train-multiple` | `1` | Train this many models from scratch and keep the best-performing one |
| `-i`, `--info` | `False` | Print a saved model's architecture, expected features, and classes |
| `-fc`, `--focus-columns` | `None` | Comma-separated feature list to restrict the *simple* model to |
| `-lb`, `--labeled` | `False` | Accepted in classify mode but currently has no effect on the output — see the note in [Classifying new, unlabeled samples](#classifying-new-unlabeled-samples) |
| `-n`, `--normalizing-function` | `none` | Normalization applied after relative-abundance scaling: `none`, `log`, `tmm` (needs `conorm`), `z-score`, `max-min`, or `stddev` |
| `-rr`, `--regex-remove` | `""` | Comma-separated regexes; matching columns are dropped before training |
| `-dr`, `--dropout` | `0.3` | Dropout rate between the network's two linear layers |
| `-it`, `--importance-thresh` | `0.5` | Minimum accuracy impact (in percent) for a feature to be kept in the simple model |
| `-wip`, `--weight-inverse-proportional` | `False` | Weight the loss inversely to class prevalence, to help with imbalanced classes |
| `-dc`, `--disable-cuda` | `False` | Force CPU even if a GPU is available |

## Tips and troubleshooting

- **"Required package X not available"** — every script checks its imports up front and exits with a plain-English message if something's missing. Match the message to the [Installation](#installation) section above.
- **Tolerance errors from `make_test_train_split.py`** — if the script complains a class's split deviates too far from the original data's balance, raise `-t` (tolerance); rare classes with few samples are the usual cause, since a single sample can swing their percentage a lot.
- **A pop-up window never appears / script seems to hang on a server** — you probably passed `--debug`. Either drop it (it's off by default) or pass `--no-debug` explicitly, and use `--test-accuracy` / `--info` for text-only diagnostics.
- **The random forest classifier is slow** — it trains 10,000 trees by default; this is intentional for accuracy, not a bug, but expect it to take meaningfully longer than the neural classifier on the same data.
- **Restoring a "simple" model** — remember `--focus-columns` only constrains the simple model, and `--no-train`/`--load` control whether a full model is (re)trained before the simple one — see the [cross-study section](#applying-a-classification-scheme-to-a-new-study) for the full sequence.
- **Checking what a model expects before feeding it new data** — always run `--info` first; mismatched feature columns between the model and your new CSV is the most common source of classify-mode errors.
