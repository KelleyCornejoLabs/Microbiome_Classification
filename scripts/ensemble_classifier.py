#!/usr/bin/env python3
"""
ensemble_classifier.py  –  Panel-of-Experts wrapper for StrataBioNN (nn_classifier.py)

Trains an ensemble of StrataBioNN single-layer networks ("experts"), each on a
(optionally bootstrapped / feature-bagged) version of the training data, then
combines their softmax outputs via soft or hard voting.

All existing nn_classifier.py flags are supported and forwarded to each expert.
New ensemble-specific flags:

  --ensemble-size  / -es   Number of expert networks          (default: 5)
  --strategy       / -str  soft_vote | hard_vote              (default: soft_vote)
  --bootstrap      / -bs   Bootstrap-sample training data per expert
  --feature-frac   / -ff   Fraction of features per expert    (default: 1.0 = all)

Usage mirrors nn_classifier.py:

  Train an ensemble:
    python3 ensemble_classifier.py -itr train.csv -ite test.csv -p my_ensemble -es 5

  Classify unlabeled data:
    python3 ensemble_classifier.py -ite unlabeled.csv -p my_ensemble -cl -out predictions

  Evaluate on labeled validation data:
    python3 ensemble_classifier.py -ite validation.csv -p my_ensemble -ta

  Bootstrap + feature-bagged ensemble:
    python3 ensemble_classifier.py -itr train.csv -ite test.csv -p ensemble_bag \\
        -es 10 -bs -ff 0.7 --strategy soft_vote

IMPORTANT: only use this approach if the standard classification produces lower than desired accuracy in the classification.
"""

import argparse
import os
import sys
import subprocess
import tempfile

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NN_CLASSIFIER = os.path.join(SCRIPT_DIR, "nn_classifier.py")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def expert_path(base: str, idx: int) -> str:
    """Return the path prefix for expert i  (nn_classifier saves <prefix>_nn.pt)."""
    return f"{base}_expert_{idx}"


def pct_columns(df: pd.DataFrame) -> list:
    """Return all 'Pct *' probability columns present in a classified output CSV."""
    return [c for c in df.columns if c.startswith("Pct ")]


def run_nn_classifier(extra_args: list, verbose: bool = True) -> int:
    """Invoke nn_classifier.py as a subprocess and return its exit code."""
    cmd = [sys.executable, NN_CLASSIFIER] + [str(a) for a in extra_args]
    if verbose:
        print(f"[ensemble] running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def bootstrap_csv(df: pd.DataFrame, seed: int, suffix: str) -> str:
    """Write a bootstrap-sampled (with replacement) version of df to a temp CSV."""
    sample = df.sample(n=len(df), replace=True, random_state=seed)
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix=f"ens_{suffix}_"
    )
    sample.to_csv(f, index=False)
    f.close()
    return f.name


def feature_bag_csv(df: pd.DataFrame, frac: float, seed: int,
                    non_feature_cols: set, suffix: str) -> tuple:
    """
    Randomly keep `frac` fraction of feature columns (everything not in
    non_feature_cols).  Returns (temp_csv_path, list_of_kept_feature_names).
    """
    features = [c for c in df.columns if c not in non_feature_cols]
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(len(features) * frac))
    chosen = list(rng.choice(features, size=n_keep, replace=False))
    keep = [c for c in df.columns if c in non_feature_cols or c in set(chosen)]
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix=f"ens_{suffix}_"
    )
    df[keep].to_csv(f, index=False)
    f.close()
    return f.name, chosen


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Panel-of-Experts ensemble wrapper for StrataBioNN (nn_classifier.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── modes (mirror nn_classifier.py) ──────────────────────────────────────
    p.add_argument("--train",            "-t",   action="store_true",  default=True,
                   help="Training mode (default)")
    p.add_argument("--no-train",                 action="store_false", dest="train",
                   help="Skip training; use a previously saved ensemble")
    p.add_argument("--classify",         "-cl",  action="store_true",  default=False,
                   help="Classify unlabeled data using a saved ensemble")
    p.add_argument("--test-accuracy",    "-ta",  action="store_true",  default=False,
                   help="Evaluate ensemble accuracy on labeled test data")

    # ── I/O (mirror nn_classifier.py) ────────────────────────────────────────
    p.add_argument("--input-train",      "-itr", default=None,
                   help="Labeled training CSV")
    p.add_argument("--input-test",       "-ite", default=None,
                   help="Test or unlabeled CSV")
    p.add_argument("--path",             "-p",   required=True,
                   help="Base path prefix for the ensemble (e.g. models/my_ensemble)")
    p.add_argument("--output",           "-out", default=None,
                   help="Output CSV prefix for ensemble predictions")

    # ── column names (mirror nn_classifier.py) ────────────────────────────────
    p.add_argument("--sample-id",        "-sid", default="sampleID",
                   help="Column name for sample IDs")
    p.add_argument("--read-count",       "-rc",  default="read_count",
                   help="Column name for read counts")
    p.add_argument("--label-col",        "-lc",  default="HC_subCST",
                   help="Column name for class labels")
    p.add_argument("--labeled",          "-lb",  action="store_true", default=False,
                   help="Treat classify-mode test data as labeled")

    # ── ensemble-specific ─────────────────────────────────────────────────────
    p.add_argument("--ensemble-size",    "-es",  type=int,   default=5,
                   help="Number of expert networks")
    p.add_argument("--strategy",         "-str", default="soft_vote",
                   choices=["soft_vote", "hard_vote"],
                   help="soft_vote: average probabilities.  hard_vote: majority class.")
    p.add_argument("--bootstrap",        "-bs",  action="store_true", default=False,
                   help="Bootstrap-sample training data for each expert")
    p.add_argument("--feature-frac",     "-ff",  type=float, default=1.0,
                   help="Fraction of features per expert (1.0 = no feature-bagging)")

    # ── forwarded nn_classifier.py hyperparameters ────────────────────────────
    p.add_argument("--learning-rate",    "-lr",  default=None)
    p.add_argument("--max-epochs",       "-me",  default=None)
    p.add_argument("--loss",             "-l",   default=None,
                   choices=["ce", "nll", "kld"])
    p.add_argument("--optim",            "-o",   default=None,
                   choices=["sgd", "adam"])
    p.add_argument("--hidden-neurons",   "-hn",  default=None)
    p.add_argument("--dropout",          "-dr",  default=None)
    p.add_argument("--patience",         "-pa",  default=None)
    p.add_argument("--threshold-lr",     "-tlr", default=None)
    p.add_argument("--metrics-interval", "-m",   default=None)
    p.add_argument("--train-multiple",   "-tm",  default=None,
                   help="Random restarts per expert (forwarded to nn_classifier.py)")
    p.add_argument("--importance-thresh","-it",  default=None)
    p.add_argument("--normalizing-function", "-n", default=None)
    p.add_argument("--regex-remove",     "-rr",  default=None)
    p.add_argument("--linear",           "-li",  action="store_true", default=False,
                   help="Use linear (no ReLU) activation in each expert")
    p.add_argument("--no-debug",                 action="store_true", default=False,
                   help="Suppress debug output from individual expert runs")

    return p


# ---------------------------------------------------------------------------
# Build the forwarded argument list for one nn_classifier.py call
# ---------------------------------------------------------------------------

def forwarded_args(args: argparse.Namespace,
                   path: str,
                   train_csv: str = None,
                   test_csv: str = None,
                   extra: list = None) -> list:
    """
    Assemble the argument list for a single nn_classifier.py invocation,
    forwarding all hyperparameters the user set.
    """
    fwd = ["-p", path]

    if train_csv:
        fwd += ["-itr", train_csv]
    elif args.input_train:
        fwd += ["-itr", args.input_train]

    if test_csv:
        fwd += ["-ite", test_csv]
    elif args.input_test:
        fwd += ["-ite", args.input_test]

#    fwd += ["-sid", args.sample_id, "-rc", args.read_count, "-lc", args.label_col]

    # Optional hyperparameters – only forward if the user actually set them
    if args.learning_rate:        fwd += ["-lr",  args.learning_rate]
    if args.max_epochs:           fwd += ["-me",  args.max_epochs]
    if args.loss:                 fwd += ["-l",   args.loss]
    if args.optim:                fwd += ["-o",   args.optim]
    if args.hidden_neurons:       fwd += ["-hn",  args.hidden_neurons]
    if args.dropout:              fwd += ["-dr",  args.dropout]
    if args.patience:             fwd += ["-pa",  args.patience]
    if args.threshold_lr:         fwd += ["-tlr", args.threshold_lr]
    if args.metrics_interval:     fwd += ["-m",   args.metrics_interval]
    if args.train_multiple:       fwd += ["-tm",  args.train_multiple]
    if args.importance_thresh:    fwd += ["-it",  args.importance_thresh]
    if args.normalizing_function: fwd += ["-n",   args.normalizing_function]
    if args.regex_remove:         fwd += ["-rr",  args.regex_remove]
    if args.linear:               fwd.append("--linear")
    if args.no_debug:             fwd.append("--no-debug")
    if args.labeled:              fwd.append("--labeled")

    if extra:
        fwd += extra

    return fwd


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ensemble(args: argparse.Namespace):
    """Train args.ensemble_size expert models."""
    train_df = pd.read_csv(args.input_train)
    non_feature_cols = {args.sample_id, args.read_count, args.label_col}

    for i in range(args.ensemble_size):
        print(f"\n{'='*60}")
        print(f"[ensemble] Training expert {i + 1} / {args.ensemble_size}")
        print(f"{'='*60}")

        ep = expert_path(args.path, i)
        seed = i * 42 + 7                         # distinct, reproducible per expert
        tmp_files = []

        working_df = train_df.copy()

        # ── optional bootstrap ────────────────────────────────────────────────
        if args.bootstrap:
            tmp = bootstrap_csv(working_df, seed=seed, suffix=f"e{i}_boot")
            tmp_files.append(tmp)
            working_df = pd.read_csv(tmp)

        # ── optional feature-bagging ──────────────────────────────────────────
        if 0.0 < args.feature_frac < 1.0:
            tmp, chosen = feature_bag_csv(
                working_df, frac=args.feature_frac,
                seed=seed, non_feature_cols=non_feature_cols,
                suffix=f"e{i}_fc"
            )
            tmp_files.append(tmp)
            train_source = tmp
            print(f"[ensemble]   feature-bagging: kept {len(chosen)} / "
                  f"{len([c for c in train_df.columns if c not in non_feature_cols])} features")
        elif args.bootstrap:
            train_source = tmp_files[0]
        else:
            train_source = args.input_train

        fwd = forwarded_args(
            args,
            path=ep,
            train_csv=train_source,
            extra=["-sd", str(seed)],
        )

        rc = run_nn_classifier(fwd, verbose=not args.no_debug)
        if rc != 0:
            print(f"[ensemble] WARNING: expert {i + 1} exited with code {rc}")

        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    print(f"\n[ensemble] All {args.ensemble_size} experts trained.")
    print(f"[ensemble] Models saved as: {args.path}_expert_{{0..{args.ensemble_size - 1}}}_nn.pt")


# ---------------------------------------------------------------------------
# Classification / evaluation
# ---------------------------------------------------------------------------

def classify_with_ensemble(args: argparse.Namespace,
                            is_labeled: bool) -> pd.DataFrame:
    """
    Run each expert in classify mode, collect output CSVs, and aggregate.
    """
    expert_dfs = []

    with tempfile.TemporaryDirectory(prefix="ens_classify_") as tmpdir:
        for i in range(args.ensemble_size):
            ep = expert_path(args.path, i)
            out_prefix = os.path.join(tmpdir, f"expert_{i}_pred")

            fwd = forwarded_args(
                args,
                path=ep,
                extra=["-cl", "-out", out_prefix] + (["-lb"] if is_labeled else []),
            )

            rc = run_nn_classifier(fwd, verbose=not args.no_debug)
            out_csv = out_prefix + ".csv"

            if rc != 0 or not os.path.exists(out_csv):
                print(f"[ensemble] WARNING: expert {i + 1} classification failed or "
                      "produced no output – skipping.")
                continue

            expert_dfs.append(pd.read_csv(out_csv))
            print(f"[ensemble] Expert {i + 1} predictions loaded.")

    if not expert_dfs:
        sys.exit("[ensemble] ERROR: No expert produced valid output.")

    return aggregate(expert_dfs, args.strategy, args.label_col, args.sample_id)


def aggregate(expert_dfs: list, strategy: str,
              label_col: str, id_col: str) -> pd.DataFrame:
    """
    Combine per-expert output DataFrames.

    soft_vote  – average the Pct columns; argmax gives final prediction.
    hard_vote  – each expert casts one vote (its argmax); majority wins.
    """
    ref = expert_dfs[0]
    pct_cols = pct_columns(ref)

    if not pct_cols:
        sys.exit(
            "[ensemble] ERROR: No 'Pct *' columns found in expert outputs. "
            "Ensure nn_classifier.py is producing probability columns."
        )

    # Stack probability matrices: shape (n_experts, n_samples, n_classes)
    prob_stack = np.stack(
        [df[pct_cols].values for df in expert_dfs],
        axis=0,
    )                                              # (E, N, C)

    avg_probs = prob_stack.mean(axis=0)            # (N, C)
    class_names = [c.replace("Pct ", "") for c in pct_cols]

    if strategy == "soft_vote":
        pred_indices = avg_probs.argmax(axis=1)

    else:  # hard_vote
        votes = prob_stack.argmax(axis=2)          # (E, N)  – each expert's choice
        # majority vote per sample
        pred_indices = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=len(class_names)).argmax(),
            axis=0,
            arr=votes,
        )

    predictions = [class_names[idx] for idx in pred_indices]

    # Assemble output in the same CSV format as nn_classifier.py classify mode:
    # sampleID | <label_col> | Pct ClassA | Pct ClassB | ...
    out = pd.DataFrame()
    if id_col in ref.columns:
        out[id_col] = ref[id_col].values
    out[label_col] = predictions
    for j, col in enumerate(pct_cols):
        out[col] = avg_probs[:, j]

    return out


# ---------------------------------------------------------------------------
# Accuracy reporting  (mirrors nn_classifier.py console style)
# ---------------------------------------------------------------------------

def report_accuracy(pred_df: pd.DataFrame, true_df: pd.DataFrame,
                    label_col: str, id_col: str):
    merged = true_df[[id_col, label_col]].merge(
        pred_df[[id_col, label_col]].rename(columns={label_col: "_pred"}),
        on=id_col,
    )
    true_labels = merged[label_col].tolist()
    pred_labels = merged["_pred"].tolist()
    classes = sorted(set(true_labels) | set(pred_labels))

    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[idx[t]][idx[p]] += 1

    correct = int(np.trace(cm))
    total = int(cm.sum())
    acc = correct / total if total else 0.0

    print(f"\n[ensemble] Accuracy: {acc:.4f}  ({correct} / {total})")
    print("\n[ensemble] Confusion matrix  (rows = true, cols = predicted):")
    print("\t" + "\t".join(classes))
    for i, c in enumerate(classes):
        print(c + "\t" + "\t".join(str(cm[i][j]) for j in range(n)))

    try:
        from sklearn.metrics import f1_score
        f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
        print(f"\n[ensemble] Weighted F1: {f1:.4f}")
    except ImportError:
        print("\n[ensemble] (install scikit-learn for weighted F1 score)")

    # Cumulative guess rankings  (same style as nn_classifier.py)
    pct_cols = pct_columns(pred_df)
    if pct_cols and id_col in pred_df.columns and id_col in true_df.columns:
        merged2 = true_df[[id_col, label_col]].merge(pred_df, on=id_col)
        class_names = [c.replace("Pct ", "") for c in pct_cols]
        correct_at = np.zeros(len(class_names), dtype=int)
        for _, row in merged2.iterrows():
            true_cls = row[label_col]
            probs = [row[c] for c in pct_cols]
            ranked = [class_names[j] for j in np.argsort(probs)[::-1]]
            for rank, guess in enumerate(ranked):
                if guess == true_cls:
                    correct_at[rank] += 1
                    break
        print("\n[ensemble] Cumulative guess rankings:")
        cumsum = 0
        for rank in range(len(class_names)):
            cumsum += correct_at[rank]
            print(f"  Top-{rank + 1}: {cumsum} / {total}  ({100 * cumsum / total:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(NN_CLASSIFIER):
        sys.exit(
            f"[ensemble] ERROR: nn_classifier.py not found at '{NN_CLASSIFIER}'.\n"
            "Ensure ensemble_classifier.py lives in the same scripts/ directory."
        )

    # ── TRAIN ────────────────────────────────────────────────────────────────
    if args.train and not args.classify and args.input_train:
        if not args.input_test:
            sys.exit("[ensemble] ERROR: --input-test / -ite required for training.")
        train_ensemble(args)

    # ── CLASSIFY / EVALUATE ───────────────────────────────────────────────────
    if args.classify or args.test_accuracy:
        if not args.input_test:
            sys.exit("[ensemble] ERROR: --input-test / -ite required for classification.")

        is_labeled = args.test_accuracy or args.labeled
        out_prefix = args.output or (args.path + "_ensemble_predictions")

        pred_df = classify_with_ensemble(args, is_labeled=is_labeled)

        out_csv = out_prefix + ".csv"
        pred_df.to_csv(out_csv, index=False)
        print(f"\n[ensemble] Ensemble predictions written to: {out_csv}")

        if args.test_accuracy:
            true_df = pd.read_csv(args.input_test)
            if args.label_col not in true_df.columns:
                print(f"[ensemble] WARNING: label column '{args.label_col}' not found "
                      "in test file – skipping accuracy report.")
            else:
                report_accuracy(pred_df, true_df, args.label_col, args.sample_id)


if __name__ == "__main__":
    main()
