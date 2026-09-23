# -*- coding: utf-8 -*-
import os
import pickle

import h5py
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from features_final_multirate import (
    extract_features,
    FEATURE_COLUMNS
)

SEED = 42

FS = 2_000_000
FRAME_SAMPLES = 1024

BASE_DIR = r"C:\Users\Admin\Desktop\digitalne"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Finalni 5-class exact-pipeline synthetic skup
SYNTHETIC_CSV = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "amc_dataset_exactpipeline_16features_2Msps_1024.csv"
)

REAL_H5 = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "subset_test.h5"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "REAL_ASSISTED_EXACTPIPELINE_16FEATURES_RESULTS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "amc_model_real_assisted_exactpipeline_16features.pkl"
)

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "real_assisted_summary.txt"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "real_assisted_predictions.csv"
)

CONFUSION_FILE = os.path.join(
    OUTPUT_DIR,
    "real_assisted_confusion_matrix.csv"
)

BY_CLASS_FILE = os.path.join(
    OUTPUT_DIR,
    "real_assisted_by_class.csv"
)

BY_SNR_FILE = os.path.join(
    OUTPUT_DIR,
    "real_assisted_by_snr.csv"
)

BY_CHANNEL_FILE = os.path.join(
    OUTPUT_DIR,
    "real_assisted_by_channel.csv"
)

IMPORTANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "real_assisted_feature_importance.csv"
)


REAL_CLASS_MAP = {
    0: "BPSK",
    1: "QPSK",
    2: "16-QAM"
}

TARGET_REAL_CLASSES = [
    "BPSK",
    "QPSK",
    "16-QAM"
]

CHANNEL_NAMES = {
    0: "Clean / LOS",
    1: "Multipath"
}

FINAL_MODEL_CLASSES = [
    "AM",
    "FM",
    "BPSK",
    "QPSK",
    "16-QAM"
]


print(
    "=" * 100
)

print(
    "REAL-ASSISTED EXACT-PIPELINE AMC - 16 OBELEZJA"
)

print(
    "=" * 100
)

print(
    "Synthetic: exact-pipeline 5-class"
)

print(
    "Real: BPSK / QPSK / 16-QAM"
)

print(
    "Real split: 70% trening / 30% test"
)

print(
    "Stratifikacija: klasa + SNR + kanal"
)

print()


print(
    "=" * 100
)

print(
    "1. UCITAVANJE EXACT-PIPELINE SINTETICKOG SKUPA"
)

print(
    "=" * 100
)


synthetic_df = pd.read_csv(
    SYNTHETIC_CSV
)


required_columns = (
    FEATURE_COLUMNS
    + [
        "label",
        "snr_db"
    ]
)


missing = [
    col
    for col in required_columns
    if col not in synthetic_df.columns
]


if missing:

    raise ValueError(
        "U sintetickom CSV-u nedostaju kolone: "
        + ", ".join(
            missing
        )
    )


if not np.isfinite(
    synthetic_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Sinteticki skup sadrzi NaN ili inf vrednosti."
    )


X_synth = synthetic_df[
    FEATURE_COLUMNS
].to_numpy(
    dtype=np.float64
)


y_synth = synthetic_df[
    "label"
].astype(
    str
).to_numpy()


print(
    "Sintetickih primera:",
    len(
        y_synth
    )
)


print(
    "\nRaspodela sintetickih klasa:"
)


for cls in FINAL_MODEL_CLASSES:

    count = int(
        np.sum(
            y_synth == cls
        )
    )

    print(
        f"{cls:7s}: {count}"
    )


if len(
    y_synth
) != 7500:

    print()

    print(
        "UPOZORENJE:"
    )

    print(
        "Rad trenutno ocekuje 7500 sintetickih primera,"
    )

    print(
        "a CSV sadrzi:",
        len(
            y_synth
        )
    )


print()

print(
    "=" * 100
)

print(
    "2. UCITAVANJE REALNOG SDR SKUPA"
)

print(
    "=" * 100
)


with h5py.File(
    REAL_H5,
    "r"
) as f:

    y_mod = f[
        "y_mod"
    ][:]

    y_chan = f[
        "y_chan"
    ][:]

    y_snr = f[
        "y_snr"
    ][:]


real_indices = np.where(
    np.isin(
        y_mod,
        [
            0,
            1,
            2
        ]
    )
)[0]


print(
    "Ukupno realnih digitalnih frejmova:",
    len(
        real_indices
    )
)


if len(
    real_indices
) != 34702:

    print(
        "UPOZORENJE: ocekivano 34702,"
        " dobijeno",
        len(
            real_indices
        )
    )


print()

print(
    "=" * 100
)

print(
    "3. EKSTRAKCIJA 16 OBELEZJA IZ REALNIH SIGNALA"
)

print(
    "=" * 100
)


real_features = []
real_labels = []
real_snrs = []
real_channels = []
real_h5_indices = []


with h5py.File(
    REAL_H5,
    "r"
) as f:

    X_real = f[
        "X"
    ]

    total = len(
        real_indices
    )

    for counter, idx in enumerate(
        real_indices
    ):

        frame = X_real[
            idx
        ].astype(
            np.float32
        )


        if frame.shape != (
            FRAME_SAMPLES,
            2
        ):

            raise ValueError(
                f"Neocekivan oblik frejma: "
                f"{frame.shape}"
            )


        iq = (
            frame[
                :,
                0
            ]
            +
            1j
            *
            frame[
                :,
                1
            ]
        ).astype(
            np.complex64
        )


        power = np.mean(
            np.abs(
                iq
            ) ** 2
        )


        if power > 1e-12:

            iq = (
                iq
                /
                np.sqrt(
                    power
                )
            ).astype(
                np.complex64
            )


        feats = extract_features(
            iq,
            fs=FS
        )


        vector = np.asarray(
            [
                feats[
                    col
                ]
                for col
                in FEATURE_COLUMNS
            ],
            dtype=np.float64
        )


        if not np.all(
            np.isfinite(
                vector
            )
        ):

            continue


        real_features.append(
            vector
        )


        real_labels.append(
            REAL_CLASS_MAP[
                int(
                    y_mod[
                        idx
                    ]
                )
            ]
        )


        real_snrs.append(
            int(
                y_snr[
                    idx
                ]
            )
        )


        real_channels.append(
            int(
                y_chan[
                    idx
                ]
            )
        )


        real_h5_indices.append(
            int(
                idx
            )
        )


        if (
            (counter + 1) % 1000 == 0
            or
            counter == total - 1
        ):

            print(
                f"Obradjeno "
                f"{counter + 1}/"
                f"{total}"
            )


real_features = np.asarray(
    real_features,
    dtype=np.float64
)

real_labels = np.asarray(
    real_labels
)

real_snrs = np.asarray(
    real_snrs,
    dtype=int
)

real_channels = np.asarray(
    real_channels,
    dtype=int
)

real_h5_indices = np.asarray(
    real_h5_indices,
    dtype=int
)


print()

print(
    "Validnih realnih frejmova:",
    len(
        real_labels
    )
)


if len(
    real_labels
) != 34702:

    print(
        "UPOZORENJE:"
        " broj validnih realnih frejmova nije 34702."
    )


stratify_key = np.asarray(
    [
        f"{label}_{snr}_{channel}"
        for label, snr, channel
        in zip(
            real_labels,
            real_snrs,
            real_channels
        )
    ]
)


# PODELA REALNOG SKUPA 70:30
print()

print(
    "=" * 100
)

print(
    "4. PODELA REALNOG SKUPA 70:30"
)

print(
    "=" * 100
)


(
    X_real_train,
    X_real_test,
    y_real_train,
    y_real_test,
    snr_train,
    snr_test,
    channel_train,
    channel_test,
    index_train,
    index_test
) = train_test_split(

    real_features,
    real_labels,
    real_snrs,
    real_channels,
    real_h5_indices,

    test_size=0.30,
    random_state=SEED,
    stratify=stratify_key
)


print(
    "Realni trening:",
    len(
        y_real_train
    )
)

print(
    "Realni test:",
    len(
        y_real_test
    )
)


if len(
    y_real_train
) != 24291:

    print(
        "UPOZORENJE:"
        " ocekivano 24291 trening primera."
    )


if len(
    y_real_test
) != 10411:

    print(
        "UPOZORENJE:"
        " ocekivano 10411 test primera."
    )


print()

print(
    "Raspodela realnog TEST skupa po klasama:"
)


for cls in TARGET_REAL_CLASSES:

    n = int(
        np.sum(
            y_real_test == cls
        )
    )

    print(
        f"{cls:7s}: {n}"
    )


print()

print(
    "Raspodela realnog TEST skupa po SNR-u:"
)


for snr in sorted(
    np.unique(
        snr_test
    )
):

    n = int(
        np.sum(
            snr_test == snr
        )
    )

    print(
        f"{snr:2d} dB: {n}"
    )


print()

print(
    "Raspodela realnog TEST skupa po kanalu:"
)


for channel in sorted(
    np.unique(
        channel_test
    )
):

    n = int(
        np.sum(
            channel_test == channel
        )
    )

    print(
        f"{CHANNEL_NAMES[channel]}: {n}"
    )


print()

print(
    "=" * 100
)

print(
    "5. FORMIRANJE KOMBINOVANOG TRENING SKUPA"
)

print(
    "=" * 100
)


X_train_combined = np.vstack(
    [
        X_synth,
        X_real_train
    ]
)


y_train_combined = np.concatenate(
    [
        y_synth,
        y_real_train
    ]
)


print(
    "Sinteticki trening primeri:",
    len(
        y_synth
    )
)

print(
    "Realni trening primeri:",
    len(
        y_real_train
    )
)

print(
    "UKUPNO trening primera:",
    len(
        y_train_combined
    )
)


if len(
    y_train_combined
) != 31791:

    print(
        "UPOZORENJE:"
    )

    print(
        "Rad trenutno ocekuje ukupno 31791 trening primer,"
    )

    print(
        "a dobijeno je:",
        len(
            y_train_combined
        )
    )


print()

print(
    "Raspodela klasa u kombinovanom treningu:"
)


combined_classes, combined_counts = np.unique(
    y_train_combined,
    return_counts=True
)


for cls, count in zip(
    combined_classes,
    combined_counts
):

    print(
        f"{cls:7s}: {count}"
    )


# RANDOM FOREST
print()

print(
    "=" * 100
)

print(
    "6. TRENIRANJE REAL-ASSISTED RANDOM FOREST MODELA"
)

print(
    "=" * 100
)


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    min_samples_split=2,
    random_state=SEED,
    n_jobs=-1
)


rf.fit(
    X_train_combined,
    y_train_combined
)


print(
    "Trening zavrsen."
)


# TEST NA ODVOJENOM REALNOM PODSKUPU
print()

print(
    "=" * 100
)

print(
    "7. TEST NA IZDVOJENOM REALNOM PODSKUPU"
)

print(
    "=" * 100
)


real_pred = rf.predict(
    X_real_test
)


overall_acc = accuracy_score(
    y_real_test,
    real_pred
)


print()

print(
    f"UKUPNA TACNOST NA IZDVOJENOM "
    f"REALNOM TESTU: "
    f"{overall_acc * 100:.2f}%"
)

print()

print(
    "=" * 100
)

print(
    "TACNOST PO KLASI"
)

print(
    "=" * 100
)


class_rows = []


for cls in TARGET_REAL_CLASSES:

    mask = (
        y_real_test
        ==
        cls
    )


    acc = accuracy_score(
        y_real_test[
            mask
        ],
        real_pred[
            mask
        ]
    )


    n = int(
        np.sum(
            mask
        )
    )


    class_rows.append(
        {
            "class":
                cls,

            "n":
                n,

            "accuracy_percent":
                acc * 100
        }
    )


    print(
        f"{cls:7s}: "
        f"{acc * 100:.2f}% "
        f"(N={n})"
    )


class_df = pd.DataFrame(
    class_rows
)


class_df.to_csv(
    BY_CLASS_FILE,
    index=False
)


print()

print(
    "=" * 100
)

print(
    "CLEAN VS MULTIPATH"
)

print(
    "=" * 100
)


channel_rows = []


for channel in sorted(
    np.unique(
        channel_test
    )
):

    mask = (
        channel_test
        ==
        channel
    )


    acc = accuracy_score(
        y_real_test[
            mask
        ],
        real_pred[
            mask
        ]
    )


    n = int(
        np.sum(
            mask
        )
    )


    channel_rows.append(
        {
            "channel":
                int(
                    channel
                ),

            "channel_name":
                CHANNEL_NAMES[
                    int(
                        channel
                    )
                ],

            "n":
                n,

            "accuracy_percent":
                acc * 100
        }
    )


    print(
        f"{CHANNEL_NAMES[channel]}: "
        f"{acc * 100:.2f}% "
        f"(N={n})"
    )


channel_df = pd.DataFrame(
    channel_rows
)


channel_df.to_csv(
    BY_CHANNEL_FILE,
    index=False
)

print()

print(
    "=" * 100
)

print(
    "TACNOST PO SNR-u"
)

print(
    "=" * 100
)


snr_rows = []


for snr in sorted(
    np.unique(
        snr_test
    )
):

    mask = (
        snr_test
        ==
        snr
    )


    acc = accuracy_score(
        y_real_test[
            mask
        ],
        real_pred[
            mask
        ]
    )


    n = int(
        np.sum(
            mask
        )
    )


    snr_rows.append(
        {
            "snr_db":
                int(
                    snr
                ),

            "n":
                n,

            "accuracy_percent":
                acc * 100
        }
    )


    print(
        f"SNR={snr:2d} dB: "
        f"{acc * 100:.2f}% "
        f"(N={n})"
    )


snr_df = pd.DataFrame(
    snr_rows
)


snr_df.to_csv(
    BY_SNR_FILE,
    index=False
)

print()

print(
    "=" * 100
)

print(
    "CONFUSION MATRIX"
)

print(
    "=" * 100
)


# U realnom testu postoje samo ove 3 stvarne klase,
# ali model ima svih 5 mogucih izlaza

prediction_labels = [
    "AM",
    "FM",
    "BPSK",
    "QPSK",
    "16-QAM"
]


cm = confusion_matrix(
    y_real_test,
    real_pred,
    labels=prediction_labels
)


cm_full_df = pd.DataFrame(
    cm,
    index=[
        f"True {x}"
        for x
        in prediction_labels
    ],
    columns=[
        f"Pred {x}"
        for x
        in prediction_labels
    ]
)


# Zadrzavamo samo redove realno prisutnih klasa
cm_real_df = cm_full_df.loc[
    [
        "True BPSK",
        "True QPSK",
        "True 16-QAM"
    ]
]


print(
    cm_real_df
)


cm_real_df.to_csv(
    CONFUSION_FILE
)


print()

print(
    "=" * 100
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 100
)


report = classification_report(
    y_real_test,
    real_pred,
    labels=TARGET_REAL_CLASSES,
    digits=4,
    zero_division=0
)


print(
    report
)


importance_df = pd.DataFrame(
    {
        "feature":
            FEATURE_COLUMNS,

        "importance":
            rf.feature_importances_
    }
).sort_values(
    "importance",
    ascending=False
)


importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


predictions_df = pd.DataFrame(
    {
        "h5_index":
            index_test,

        "true_label":
            y_real_test,

        "predicted_label":
            real_pred,

        "snr_db":
            snr_test,

        "channel":
            channel_test
    }
)


predictions_df[
    "channel_name"
] = predictions_df[
    "channel"
].map(
    CHANNEL_NAMES
)


predictions_df[
    "correct"
] = (
    predictions_df[
        "true_label"
    ]
    ==
    predictions_df[
        "predicted_label"
    ]
)


predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

with open(
    MODEL_FILE,
    "wb"
) as f:

    pickle.dump(
        {
            "model":
                rf,

            "feature_columns":
                FEATURE_COLUMNS,

            "fs":
                FS,

            "frame_samples":
                FRAME_SAMPLES,

            "classes":
                FINAL_MODEL_CLASSES,

            "real_target_classes":
                TARGET_REAL_CLASSES,

            "synthetic_dataset":
                SYNTHETIC_CSV,

            "real_dataset":
                REAL_H5,

            "real_split":
                "70:30",

            "stratification":
                "class + SNR + channel",

            "seed":
                SEED,

            "training_type":
                "real_assisted_exactpipeline_16features"
        },
        f
    )


with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "REAL-ASSISTED EXACT-PIPELINE "
        "16-FEATURE EXPERIMENT\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        f"Synthetic examples: "
        f"{len(y_synth)}\n"
    )

    f.write(
        f"Real total examples: "
        f"{len(real_labels)}\n"
    )

    f.write(
        f"Real training examples: "
        f"{len(y_real_train)}\n"
    )

    f.write(
        f"Real test examples: "
        f"{len(y_real_test)}\n"
    )

    f.write(
        f"Combined training examples: "
        f"{len(y_train_combined)}\n\n"
    )

    f.write(
        f"Overall real test accuracy: "
        f"{overall_acc * 100:.4f}%\n\n"
    )

    f.write(
        "ACCURACY BY CLASS\n"
    )

    f.write(
        class_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nACCURACY BY CHANNEL\n"
    )

    f.write(
        channel_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nACCURACY BY SNR\n"
    )

    f.write(
        snr_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nCONFUSION MATRIX\n"
    )

    f.write(
        cm_real_df.to_string()
    )

    f.write(
        "\n\nCLASSIFICATION REPORT\n"
    )

    f.write(
        report
    )

    f.write(
        "\n\nFEATURE IMPORTANCE\n"
    )

    f.write(
        importance_df.to_string(
            index=False
        )
    )

print()

print(
    "=" * 100
)

print(
    "ZAVRSENO"
)

print(
    "=" * 100
)


print(
    "Synthetic:",
    len(
        y_synth
    )
)

print(
    "Real trening:",
    len(
        y_real_train
    )
)

print(
    "Real test:",
    len(
        y_real_test
    )
)

print(
    "Kombinovani trening:",
    len(
        y_train_combined
    )
)

print(
    f"Finalna real-assisted tacnost: "
    f"{overall_acc * 100:.2f}%"
)


print()

print(
    "Sacuvano u:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "- real_assisted_summary.txt"
)

print(
    "- real_assisted_by_class.csv"
)

print(
    "- real_assisted_by_channel.csv"
)

print(
    "- real_assisted_by_snr.csv"
)

print(
    "- real_assisted_confusion_matrix.csv"
)

print(
    "- real_assisted_predictions.csv"
)

print(
    "- real_assisted_feature_importance.csv"
)

print(
    "- amc_model_real_assisted_exactpipeline_16features.pkl"
)