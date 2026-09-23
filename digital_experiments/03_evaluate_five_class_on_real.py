# -*- coding: utf-8 -*-

import os
import pickle
import h5py
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from features_final_multirate import extract_features, FEATURE_COLUMNS


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "amc_model_exactpipeline_16features_2Msps_1024.pkl"
)

REAL_H5 = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "subset_test.h5"
)
OUTPUT_DIR = "REAL_DIGITAL_EXACTPIPELINE_RESULTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REAL_FS = 2_000_000

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
    0: "Clean / LOS",
    1: "Multipath",
}


with open(MODEL_FILE, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
MODEL_FEATURE_COLUMNS = list(bundle["feature_columns"])
MODEL_FS = int(bundle["fs"])
MODEL_SAMPLES = int(bundle["final_samples"])

if MODEL_FEATURE_COLUMNS != FEATURE_COLUMNS:
    raise ValueError("Feature kolone modela i extractor-a se ne poklapaju.")

if MODEL_FS != REAL_FS:
    raise ValueError(
        f"Model fs={MODEL_FS}, realni dataset fs={REAL_FS}."
    )

if MODEL_SAMPLES != 1024:
    raise ValueError(
        f"Model ocekuje {MODEL_SAMPLES} uzoraka, a realni frejm ima 1024."
    )


print("=" * 100)
print("REALNI DIGITALNI SIGNALI - EXACT-PIPELINE 16-FEATURE MODEL")
print("=" * 100)
print("Model:", MODEL_FILE)
print("Broj obelezja:", len(MODEL_FEATURE_COLUMNS))
print("Klase modela:", list(model.classes_))
print()


with h5py.File(REAL_H5, "r") as f:
    y_mod = f["y_mod"][:]
    y_chan = f["y_chan"][:]
    y_snr = f["y_snr"][:]

indices = np.where(
    np.isin(y_mod, [0, 1, 2])
)[0]

print(
    "Ukupno realnih BPSK/QPSK/16-QAM frejmova:",
    len(indices)
)

features = []
labels = []
snrs = []
channels = []
h5_indices = []


with h5py.File(REAL_H5, "r") as f:

    X = f["X"]
    total = len(indices)

    for counter, idx in enumerate(indices):

        frame = X[idx].astype(np.float32)

        if frame.shape != (1024, 2):
            raise ValueError(
                f"Neocekivan oblik frejma: {frame.shape}"
            )

        iq = (
            frame[:, 0]
            + 1j * frame[:, 1]
        ).astype(np.complex64)

        power = np.mean(np.abs(iq) ** 2)

        if power > 1e-12:
            iq = iq / np.sqrt(power)

        feats = extract_features(
            iq,
            fs=REAL_FS
        )

        vector = np.array(
            [
                feats[col]
                for col in MODEL_FEATURE_COLUMNS
            ],
            dtype=np.float64
        )

        if not np.all(np.isfinite(vector)):
            continue

        features.append(vector)
        labels.append(
            REAL_CLASS_MAP[int(y_mod[idx])]
        )
        snrs.append(int(y_snr[idx]))
        channels.append(int(y_chan[idx]))
        h5_indices.append(int(idx))

        if (
            (counter + 1) % 1000 == 0
            or counter == total - 1
        ):
            print(
                f"Obradjeno {counter + 1}/{total}"
            )


features = np.asarray(features, dtype=np.float64)
labels = np.asarray(labels)
snrs = np.asarray(snrs)
channels = np.asarray(channels)
h5_indices = np.asarray(h5_indices)

X_real = pd.DataFrame(
    features,
    columns=MODEL_FEATURE_COLUMNS
)

pred = model.predict(X_real)
proba = model.predict_proba(X_real)

overall_acc = accuracy_score(labels, pred)

print()
print("=" * 100)
print("UKUPNI REZULTAT")
print("=" * 100)
print(
    f"UKUPNA REAL-WORLD TACNOST: "
    f"{overall_acc * 100:.2f}%"
)


print()
print("TACNOST PO KLASI")

class_rows = []

for cls in TARGET_CLASSES:

    mask = labels == cls

    acc = accuracy_score(
        labels[mask],
        pred[mask]
    )

    n = int(np.sum(mask))

    class_rows.append({
        "class": cls,
        "accuracy_percent": acc * 100,
        "n": n,
    })

    print(
        f"{cls}: "
        f"{acc * 100:.2f}% "
        f"(N={n})"
    )


print()
print("TACNOST PO SNR-u")

snr_rows = []

for snr in sorted(np.unique(snrs)):

    mask = snrs == snr

    acc = accuracy_score(
        labels[mask],
        pred[mask]
    )

    n = int(np.sum(mask))

    snr_rows.append({
        "snr_db": int(snr),
        "accuracy_percent": acc * 100,
        "n": n,
    })

    print(
        f"SNR={snr:2d} dB: "
        f"{acc * 100:.2f}% "
        f"(N={n})"
    )


print()
print("CLEAN VS MULTIPATH")

channel_rows = []

for ch in sorted(np.unique(channels)):

    mask = channels == ch

    acc = accuracy_score(
        labels[mask],
        pred[mask]
    )

    n = int(np.sum(mask))

    channel_rows.append({
        "channel": int(ch),
        "channel_name": CHANNEL_NAMES[int(ch)],
        "accuracy_percent": acc * 100,
        "n": n,
    })

    print(
        f"{CHANNEL_NAMES[int(ch)]}: "
        f"{acc * 100:.2f}% "
        f"(N={n})"
    )


print()
print("CONFUSION MATRIX")

all_classes = list(model.classes_)

cm = confusion_matrix(
    labels,
    pred,
    labels=all_classes
)

cm_df = pd.DataFrame(
    cm,
    index=[f"True {x}" for x in all_classes],
    columns=[f"Pred {x}" for x in all_classes],
)

print(cm_df)


print()
print("CLASSIFICATION REPORT")

report = classification_report(
    labels,
    pred,
    labels=TARGET_CLASSES,
    digits=4,
    zero_division=0,
)

print(report)


results_df = pd.DataFrame({
    "h5_index": h5_indices,
    "true_label": labels,
    "predicted_label": pred,
    "snr_db": snrs,
    "channel": channels,
})

results_df["channel_name"] = results_df["channel"].map(CHANNEL_NAMES)

for i, cls in enumerate(model.classes_):
    safe = cls.replace("-", "").replace(" ", "_")
    results_df[f"p_{safe}"] = proba[:, i]

sorted_probs = np.sort(proba, axis=1)

results_df["p_max"] = sorted_probs[:, -1]
results_df["margin"] = (
    sorted_probs[:, -1]
    - sorted_probs[:, -2]
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    ),
    index=False
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)

pd.DataFrame(class_rows).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "by_class.csv"
    ),
    index=False
)

pd.DataFrame(snr_rows).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "by_snr.csv"
    ),
    index=False
)

pd.DataFrame(channel_rows).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "by_channel.csv"
    ),
    index=False
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "summary.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"UKUPNA REAL-WORLD TACNOST: "
        f"{overall_acc * 100:.4f}%\n\n"
    )

    f.write("TACNOST PO KLASI\n")

    for row in class_rows:
        f.write(
            f"{row['class']}: "
            f"{row['accuracy_percent']:.4f}% "
            f"(N={row['n']})\n"
        )

    f.write("\nTACNOST PO SNR-u\n")

    for row in snr_rows:
        f.write(
            f"{row['snr_db']} dB: "
            f"{row['accuracy_percent']:.4f}% "
            f"(N={row['n']})\n"
        )

    f.write("\nCLEAN VS MULTIPATH\n")

    for row in channel_rows:
        f.write(
            f"{row['channel_name']}: "
            f"{row['accuracy_percent']:.4f}% "
            f"(N={row['n']})\n"
        )

    f.write("\nCONFUSION MATRIX\n")
    f.write(cm_df.to_string())

    f.write("\n\nCLASSIFICATION REPORT\n")
    f.write(report)

print()
print("Sacuvano u:", OUTPUT_DIR)
