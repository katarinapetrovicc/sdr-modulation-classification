# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "00_osnovni_sinteticki_skup_16_obelezja.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "base_classification"
    / "rf_overfitting_check"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RANDOM_STATE = 42

FEATURE_NAMES = [
    "amp_var",
    "amp_kurtosis",
    "amp_skew",
    "phase_var",
    "phase_kurtosis",
    "freq_var",
    "freq_kurtosis",
    "center_power",
    "spectral_flatness",
    "C20_abs",
    "C21_abs",
    "C40_abs",
    "C42_abs",
    "carrier_power_ratio",
    "sideband_symmetry",
    "envelope_modulation_index"
]


df = pd.read_csv(
    DATA_FILE
)

X = df[
    FEATURE_NAMES
].values

y = df[
    "class"
].values


print("=" * 90)
print("RANDOM FOREST - KONTROLA OVERFITTING-a")
print("=" * 90)

print(
    f"Broj primera: {len(y)}"
)

print(
    f"Broj obelezja: {X.shape[1]}"
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


configurations = []

for max_depth in [
    10,
    12,
    15,
    18
]:

    for min_samples_leaf in [
        2,
        4,
        6
    ]:

        configurations.append({
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf
        })

configurations.append({
    "max_depth": 20,
    "min_samples_leaf": 2
})


# 10-FOLD CV

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)


results = []


for i, config in enumerate(
    configurations,
    start=1
):

    print()
    print("-" * 90)

    print(
        f"Konfiguracija {i}/{len(configurations)}"
    )

    print(
        f"max_depth = {config['max_depth']}, "
        f"min_samples_leaf = {config['min_samples_leaf']}"
    )


    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=config[
            "max_depth"
        ],
        min_samples_split=2,
        min_samples_leaf=config[
            "min_samples_leaf"
        ],
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    train_pred = model.predict(
        X_train
    )

    test_pred = model.predict(
        X_test
    )


    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )


    gap_pp = (
        train_accuracy
        - test_accuracy
    ) * 100.0

    # 10-FOLD CV

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )


    cv_mean = np.mean(
        cv_scores
    )

    cv_std = np.std(
        cv_scores,
        ddof=1
    )


    print(
        f"Train: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Test: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Train-test razlika: "
        f"{gap_pp:.2f} p.p."
    )

    print(
        f"CV: "
        f"{cv_mean * 100:.2f}% "
        f"± {cv_std * 100:.2f} p.p."
    )


    results.append({
        "max_depth":
            config[
                "max_depth"
            ],

        "min_samples_leaf":
            config[
                "min_samples_leaf"
            ],

        "train_accuracy":
            train_accuracy,

        "test_accuracy":
            test_accuracy,

        "train_test_gap_pp":
            gap_pp,

        "cv_mean_accuracy":
            cv_mean,

        "cv_std_accuracy":
            cv_std,

        "cv_min_accuracy":
            np.min(
                cv_scores
            ),

        "cv_max_accuracy":
            np.max(
                cv_scores
            )
    })


results_df = pd.DataFrame(
    results
)


# sortira se prvenstveno po CV tacnosti,
# zatim po test tacnosti,
# a zatim po manjem train-test razlici

results_df = results_df.sort_values(
    by=[
        "cv_mean_accuracy",
        "test_accuracy",
        "train_test_gap_pp"
    ],
    ascending=[
        False,
        False,
        True
    ]
).reset_index(
    drop=True
)


CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "RF_overfitting_check.csv"
)

results_df.to_csv(
    CSV_PATH,
    index=False
)


print()
print("=" * 90)
print("RANGIRANI REZULTATI")
print("=" * 90)

display_df = results_df.copy()

display_df[
    "train_accuracy"
] *= 100.0

display_df[
    "test_accuracy"
] *= 100.0

display_df[
    "cv_mean_accuracy"
] *= 100.0

display_df[
    "cv_std_accuracy"
] *= 100.0

display_df[
    "cv_min_accuracy"
] *= 100.0

display_df[
    "cv_max_accuracy"
] *= 100.0


print(
    display_df[
        [
            "max_depth",
            "min_samples_leaf",
            "train_accuracy",
            "test_accuracy",
            "train_test_gap_pp",
            "cv_mean_accuracy",
            "cv_std_accuracy"
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)

best = results_df.iloc[
    0
]


print()
print("=" * 90)
print("NAJBOLJA KONFIGURACIJA PO CV TACNOSTI")
print("=" * 90)

print(
    f"max_depth = "
    f"{int(best['max_depth'])}"
)

print(
    f"min_samples_leaf = "
    f"{int(best['min_samples_leaf'])}"
)

print(
    f"Train accuracy = "
    f"{best['train_accuracy'] * 100:.2f}%"
)

print(
    f"Test accuracy = "
    f"{best['test_accuracy'] * 100:.2f}%"
)

print(
    f"Train-test gap = "
    f"{best['train_test_gap_pp']:.2f} p.p."
)

print(
    f"CV accuracy = "
    f"{best['cv_mean_accuracy'] * 100:.2f}% "
    f"± "
    f"{best['cv_std_accuracy'] * 100:.2f} p.p."
)

labels = [
    (
        f"d={int(row.max_depth)}, "
        f"leaf={int(row.min_samples_leaf)}"
    )
    for row in results_df.itertuples()
]

x_pos = np.arange(
    len(results_df)
)


plt.figure(
    figsize=(13, 7)
)

plt.plot(
    x_pos,
    results_df[
        "test_accuracy"
    ] * 100.0,
    marker="o",
    linewidth=2,
    label="Тест тачност"
)

plt.plot(
    x_pos,
    results_df[
        "cv_mean_accuracy"
    ] * 100.0,
    marker="s",
    linewidth=2,
    label="Просечна CV тачност"
)

plt.xticks(
    x_pos,
    labels,
    rotation=45,
    ha="right"
)

plt.ylabel(
    "Тачност [%]"
)

plt.xlabel(
    "Конфигурација Random Forest модела"
)

plt.title(
    "Поређење тест и CV тачности Random Forest конфигурација"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_RF_test_i_CV_tacnost.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(
    figsize=(13, 7)
)

bars = plt.bar(
    x_pos,
    results_df[
        "train_test_gap_pp"
    ]
)

plt.xticks(
    x_pos,
    labels,
    rotation=45,
    ha="right"
)

plt.ylabel(
    "Разлика train–test [процентни поени]"
)

plt.xlabel(
    "Конфигурација Random Forest модела"
)

plt.title(
    "Разлика између тренинг и тест тачности"
)

plt.grid(
    axis="y",
    alpha=0.3
)


for bar, value in zip(
    bars,
    results_df[
        "train_test_gap_pp"
    ]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2.0,
        bar.get_height()
        + 0.15,
        f"{value:.2f}",
        ha="center",
        fontsize=8
    )


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_RF_train_test_gap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("=" * 90)
print("SACUVANO")
print("=" * 90)

print(
    CSV_PATH
)

print(
    OUTPUT_DIR
)

print()
print("GOTOVO.")