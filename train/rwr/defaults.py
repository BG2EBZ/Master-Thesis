from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from train.common.evaluation_seeds import FIXED_EVALUATION_SEEDS

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"

DEFAULT_EPOCHS = 100
DEFAULT_SAMPLES_PER_EPOCH = 50
DEFAULT_SEED = 42
DEFAULT_BETA = 0.1
DEFAULT_EPOCH_TRAIN_SEED_COUNT = 1
DEFAULT_N_HUMANS = 15
DEFAULT_MAX_WORKERS = 10
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "runs" / f"rwr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_CSV_NAME = "training_metrics.csv"
DEFAULT_PLOT_NAME = "training_metrics.png"
DEFAULT_EXPLORATION_PLOT_NAME = "exploration_metrics.png"
DEFAULT_BEST_PARAMS_NAME = "best_params.json"
DEFAULT_LEARNING_CURVE_RAW_CSV_NAME = "learning_curve_raw.csv"
DEFAULT_LEARNING_CURVE_MATRIX_CSV_NAME = "learning_curve_matrix.csv"
DEFAULT_LEARNING_CURVE_PLOT_NAME = "learning_curve_plot.png"
DEFAULT_LEARNING_CURVE_SUMMARY_NAME = "learning_curve_summary.json"
DEFAULT_N_LEARNING_SEEDS = 20
DEFAULT_N_EVAL_SEEDS = min(20, len(FIXED_EVALUATION_SEEDS))

METRIC_FIELDNAMES = (
    "epoch",
    "mean_return",
    "best_return",
    "std_0",
    "std_1",
    "std_2",
    "std_3",
    "std_4",
    "distribution_entropy",
    "mean_duration_seconds",
    "mean_overwhelmed_triggers",
    "mean_impatient_triggers",
    "mean_distracted_triggers",
)
LEARNING_CURVE_RAW_FIELDNAMES = (
    "policy",
    "learning_seed",
    "evaluation_seed",
    "epoch",
    "mean_return",
    "mean_duration_seconds",
    "mean_overwhelmed_triggers",
    "mean_impatient_triggers",
    "mean_distracted_triggers",
)

INITIAL_MU = np.array(
    [
        2.5,
        3.5,
        2.0,
        0.7,
        1.0,
    ],
    dtype=np.float64,
)
INITIAL_STD = np.array(
    [
        0.4,
        0.6,
        0.8,
        0.1,
        0.05,
    ],
    dtype=np.float64,
)
