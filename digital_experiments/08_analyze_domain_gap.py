# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import pandas as pd

from features_final_multirate import (
    extract_features,
    FEATURE_COLUMNS
)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    "DOMAIN_GAP_DIAGNOSTICS_FINAL"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


FS = 2_000_000

REAL_CLASS_MAP = {
    0: "BPSK",
    1: "QPSK",
    2: "16-QAM",
}

TARGET_CLASSES = [
    "BPSK",
    "QPSK",
    "16-QAM",
]

CHANNEL_NAMES = {
    0: "Clean_LOS",
    1: "Multipath",
}


def safe_iqr(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    q25 = np.percentile(
        x,
        25
    )

    q75 = np.percentile(
        x,
        75
    )

    return float(
        q75 - q25
    )


def robust_gap(
    synth_values,
    real_values
):

    synth_values = np.asarray(
        synth_values,
        dtype=np.float64
    )

    real_values = np.asarray(
        real_values,
        dtype=np.float64
    )

    med_s = float(
        np.median(
            synth_values
        )
    )

    med_r = float(
        np.median(
            real_values
        )
    )

    iqr_s = safe_iqr(
        synth_values
    )

    iqr_r = safe_iqr(
        real_values
    )

    robust_scale = 0.5 * (
        iqr_s + iqr_r
    )

    gap = (
        abs(
            med_r - med_s
        )
        /
        (
            robust_scale
            + 1e-12
        )
    )

    return {
        "synthetic_mean":
            float(
                np.mean(
                    synth_values
                )
            ),

        "real_mean":
            float(
                np.mean(
                    real_values
                )
            ),

        "synthetic_median":
            med_s,

        "real_median":
            med_r,

        "synthetic_std":
            float(
                np.std(
                    synth_values
                )
            ),

        "real_std":
            float(
                np.std(
                    real_values
                )
            ),

        "synthetic_iqr":
            iqr_s,

        "real_iqr":
            iqr_r,

        "robust_gap":
            float(
                gap
            ),
    }

print(
    "=" * 100
)

print(
    "DOMAIN GAP DIJAGNOSTIKA - FINALNA VERZIJA"
)

print(
    "EXACT-PIPELINE SYNTHETIC VS SVI REALNI DIGITALNI FREJMOVI"
)

print(
    "=" * 100
)


# UCITAVANJE SINTETICKOG SKUPA

print()
print(
    "=" * 100
)
print(
    "UCITAVANJE EXACT-PIPELINE SINTETICKOG SKUPA"
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


# Zadrzavamo samo digitalne klase

synthetic_df = synthetic_df[
    synthetic_df[
        "label"
    ].isin(
        TARGET_CLASSES
    )
].copy()


print(
    "Sinteticki digitalni primeri:",
    len(
        synthetic_df
    )
)

print(
    synthetic_df[
        "label"
    ].value_counts()
)

print()


print(
    "=" * 100
)

print(
    "UCITAVANJE REALNOG SDR SKUPA"
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


selected_indices = []

print()

print(
    "=" * 100
)

print(
    "BIRANJE SVIH REALNIH DIGITALNIH FREJMOVA"
)

print(
    "=" * 100
)


for mod_id, cls in REAL_CLASS_MAP.items():

    for channel_id, channel_name in CHANNEL_NAMES.items():

        candidates = np.where(
            (y_mod == mod_id)
            &
            (y_chan == channel_id)
        )[0]

        # Uzimamo SVE frejmove
        chosen = candidates

        selected_indices.extend(
            chosen.tolist()
        )

        print(
            f"{cls:7s} | "
            f"{channel_name:10s} | "
            f"uzeto {len(chosen)}"
        )


selected_indices = np.asarray(
    selected_indices,
    dtype=np.int64
)


print()

print(
    "Ukupno realnih digitalnih frejmova:",
    len(
        selected_indices
    )
)

print()


expected_total = 34702

if len(
    selected_indices
) != expected_total:

    print(
        "UPOZORENJE:"
    )

    print(
        "Ocekivano je",
        expected_total,
        "frejmova, a pronadjeno",
        len(
            selected_indices
        )
    )

else:

    print(
        "Broj realnih frejmova je ocekivan:",
        expected_total
    )

print()

real_rows = []

print(
    "=" * 100
)

print(
    "EKSTRAKCIJA REALNIH 16 OBELEZJA"
)

print(
    "=" * 100
)


with h5py.File(
    REAL_H5,
    "r"
) as f:

    X = f[
        "X"
    ]

    total = len(
        selected_indices
    )

    for counter, idx in enumerate(
        selected_indices
    ):

        frame = X[
            idx
        ].astype(
            np.float32
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

        # Normalizacija snage
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
            )

        # Ekstrakcija istih 16 obelezja
        feats = extract_features(
            iq,
            fs=FS
        )

        # Provera numerickih vrednosti
        valid = all(
            np.isfinite(
                feats[
                    col
                ]
            )
            for col in FEATURE_COLUMNS
        )

        if not valid:

            continue

        row = {
            feature:
                feats[
                    feature
                ]

            for feature
            in FEATURE_COLUMNS
        }

        row[
            "label"
        ] = REAL_CLASS_MAP[
            int(
                y_mod[
                    idx
                ]
            )
        ]

        row[
            "channel"
        ] = CHANNEL_NAMES[
            int(
                y_chan[
                    idx
                ]
            )
        ]

        row[
            "snr_db"
        ] = int(
            y_snr[
                idx
            ]
        )

        row[
            "h5_index"
        ] = int(
            idx
        )

        real_rows.append(
            row
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


real_df = pd.DataFrame(
    real_rows
)


print()

print(
    "Validnih realnih frejmova:",
    len(
        real_df
    )
)

print()


real_features_path = os.path.join(
    OUTPUT_DIR,
    "real_features_all_34702_16features.csv"
)

real_df.to_csv(
    real_features_path,
    index=False
)

comparison_rows = []

print(
    "=" * 100
)

print(
    "RACUNANJE SYNTHETIC-TO-REAL ROBUST GAP-A"
)

print(
    "=" * 100
)


for cls in TARGET_CLASSES:

    synth_cls = synthetic_df[
        synthetic_df[
            "label"
        ]
        ==
        cls
    ]

    for channel_name in [
        "Clean_LOS",
        "Multipath"
    ]:

        real_cls = real_df[
            (
                real_df[
                    "label"
                ]
                ==
                cls
            )
            &
            (
                real_df[
                    "channel"
                ]
                ==
                channel_name
            )
        ]

        print(
            f"{cls:7s} | "
            f"{channel_name:10s} | "
            f"Synthetic={len(synth_cls)} | "
            f"Real={len(real_cls)}"
        )

        for feature in FEATURE_COLUMNS:

            stats = robust_gap(
                synth_cls[
                    feature
                ].to_numpy(),

                real_cls[
                    feature
                ].to_numpy()
            )

            comparison_rows.append(
                {
                    "class":
                        cls,

                    "channel":
                        channel_name,

                    "feature":
                        feature,

                    **stats,
                }
            )


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df = comparison_df.sort_values(
    [
        "class",
        "channel",
        "robust_gap"
    ],
    ascending=[
        True,
        True,
        False
    ]
)


comparison_path = os.path.join(
    OUTPUT_DIR,
    "feature_domain_gap_full.csv"
)


comparison_df.to_csv(
    comparison_path,
    index=False
)

top_rows = []


print()

print(
    "=" * 100
)

print(
    "NAJVECE RAZLIKE SYNTHETIC -> REAL"
)

print(
    "=" * 100
)


for cls in TARGET_CLASSES:

    for channel_name in [
        "Clean_LOS",
        "Multipath"
    ]:

        subset = comparison_df[
            (
                comparison_df[
                    "class"
                ]
                ==
                cls
            )
            &
            (
                comparison_df[
                    "channel"
                ]
                ==
                channel_name
            )
        ].sort_values(
            "robust_gap",
            ascending=False
        )

        top = subset.head(
            6
        )

        print()
        print(
            f"{cls} | "
            f"{channel_name}"
        )

        print(
            top[
                [
                    "feature",
                    "synthetic_median",
                    "real_median",
                    "robust_gap"
                ]
            ].to_string(
                index=False
            )
        )

        top_rows.append(
            top
        )


top_df = pd.concat(
    top_rows,
    ignore_index=True
)


top_path = os.path.join(
    OUTPUT_DIR,
    "feature_domain_gap_top6.csv"
)


top_df.to_csv(
    top_path,
    index=False
)

global_rank = (
    comparison_df
    .groupby(
        "feature",
        as_index=False
    )[
        "robust_gap"
    ]
    .mean()
    .sort_values(
        "robust_gap",
        ascending=False
    )
)


global_path = os.path.join(
    OUTPUT_DIR,
    "feature_domain_gap_global_ranking.csv"
)


global_rank.to_csv(
    global_path,
    index=False
)


print()

print(
    "=" * 100
)

print(
    "GLOBALNO NAJPROBLEMATICNIJA OBELEZJA"
)

print(
    "=" * 100
)


print(
    global_rank.to_string(
        index=False
    )
)


channel_rank = (
    comparison_df
    .groupby(
        [
            "channel",
            "feature"
        ],
        as_index=False
    )[
        "robust_gap"
    ]
    .mean()
)


channel_rank = channel_rank.sort_values(
    [
        "channel",
        "robust_gap"
    ],
    ascending=[
        True,
        False
    ]
)


channel_rank_path = os.path.join(
    OUTPUT_DIR,
    "feature_domain_gap_by_channel.csv"
)


channel_rank.to_csv(
    channel_rank_path,
    index=False
)


print()

print(
    "=" * 100
)

print(
    "GLOBALNI GAP PO TIPU KANALA"
)

print(
    "=" * 100
)


for channel_name in [
    "Clean_LOS",
    "Multipath"
]:

    print()
    print(
        channel_name
    )

    tmp = channel_rank[
        channel_rank[
            "channel"
        ]
        ==
        channel_name
    ].head(
        8
    )

    print(
        tmp[
            [
                "feature",
                "robust_gap"
            ]
        ].to_string(
            index=False
        )
    )

summary_path = os.path.join(
    OUTPUT_DIR,
    "domain_gap_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "DOMAIN GAP DIJAGNOSTIKA - FINALNA VERZIJA\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        "Synthetic skup: "
        "exact-pipeline 2 MS/s, 1024 uzorka\n"
    )

    f.write(
        "Analizirane klase: "
        "BPSK, QPSK, 16-QAM\n"
    )

    f.write(
        f"Sinteticki digitalni primeri: "
        f"{len(synthetic_df)}\n"
    )

    f.write(
        f"Realni analizirani frejmovi: "
        f"{len(real_df)}\n\n"
    )

    f.write(
        "ROBUST GAP DEFINICIJA:\n"
    )

    f.write(
        "|median_real - median_synthetic| "
        "/ [0.5 * (IQR_synthetic + IQR_real)]\n\n"
    )

    for cls in TARGET_CLASSES:

        for channel_name in [
            "Clean_LOS",
            "Multipath"
        ]:

            subset = comparison_df[
                (
                    comparison_df[
                        "class"
                    ]
                    ==
                    cls
                )
                &
                (
                    comparison_df[
                        "channel"
                    ]
                    ==
                    channel_name
                )
            ].sort_values(
                "robust_gap",
                ascending=False
            )

            f.write(
                "\n"
                +
                cls
                +
                " | "
                +
                channel_name
                +
                "\n"
            )

            f.write(
                subset.head(
                    6
                )[
                    [
                        "feature",
                        "synthetic_median",
                        "real_median",
                        "robust_gap"
                    ]
                ].to_string(
                    index=False
                )
            )

            f.write(
                "\n"
            )

    f.write(
        "\n\nGLOBALNI RANG:\n"
    )

    f.write(
        global_rank.to_string(
            index=False
        )
    )

    f.write(
        "\n\nGLOBALNI RANG PO TIPU KANALA:\n"
    )

    f.write(
        channel_rank.to_string(
            index=False
        )
    )

print()

print(
    "=" * 100
)

print(
    "SACUVANO"
)

print(
    "=" * 100
)


print(
    "-",
    real_features_path
)

print(
    "-",
    comparison_path
)

print(
    "-",
    top_path
)

print(
    "-",
    global_path
)

print(
    "-",
    channel_rank_path
)

print(
    "-",
    summary_path
)


print()

print(
    "GOTOVO."
)

print(
    "Za poglavlje 6.8.4 posalji:"
)

print(
    "1) feature_domain_gap_global_ranking.csv"
)

print(
    "2) feature_domain_gap_full.csv"
)

print(
    "3) domain_gap_summary.txt"
)