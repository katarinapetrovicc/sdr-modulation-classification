import pickle
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features_final import extract_features, FEATURE_COLUMNS


SEED = 42
rng = np.random.default_rng(SEED)

FS = 500_000
RAW_DURATION = 0.012
FINAL_DURATION = 0.010
RAW_SAMPLES = int(FS * RAW_DURATION)
FINAL_SAMPLES = int(FS * FINAL_DURATION)
SYMBOL_RATE = 10_000

SNR_VALUES = [-10, -5, 0, 5, 10, 15, 20]
CLASSES = ["AM", "FM", "BPSK", "QPSK", "16-QAM"]
EXAMPLES_PER_CLASS_PER_SNR = 250

DATASET_NAME = "amc_dataset_final_16features_500ksps_10ms.csv"
MODEL_NAME = "amc_model_final_16features_500ksps_10ms.pkl"
RESULTS_NAME = "amc_model_final_16features_results.txt"
IMPORTANCE_NAME = "amc_model_final_16features_importance.csv"

FIR = signal.firwin(129, 100_000, fs=FS)


def add_awgn(iq, snr_db):
    sig_power = np.mean(np.abs(iq) ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
    noise *= np.sqrt(noise_power / 2.0)
    return (iq + noise).astype(np.complex64)


def rrc_taps(beta, sps, span_symbols=8):
    n_taps = span_symbols * sps + 1
    t = (np.arange(n_taps) - (n_taps - 1) / 2) / sps
    h = np.zeros(n_taps, dtype=np.float64)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-10:
            h[i] = (
                beta / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            )
        else:
            num = (
                np.sin(np.pi * ti * (1.0 - beta))
                + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den

    h /= np.sqrt(np.sum(h ** 2) + 1e-12)
    return h


def shape_symbols(symbols, sps):
    beta = rng.uniform(0.20, 0.50)
    taps = rrc_taps(beta, sps)
    return signal.upfirdn(taps, symbols, up=sps).astype(np.complex64)


def generate_am(n_samples):
    """Raznovrstan AM-DSB-TC, bez prilagodjavanja jednom snimku."""
    t = np.arange(n_samples) / FS

    white = rng.standard_normal(n_samples + 512)
    taps1 = signal.firwin(257, [250.0, 3600.0], pass_zero=False, fs=FS)
    speech = signal.lfilter(taps1, 1.0, white)[256:256 + n_samples]

    white2 = rng.standard_normal(n_samples + 512)
    taps2 = signal.firwin(257, [200.0, 1800.0], pass_zero=False, fs=FS)
    speech_low = signal.lfilter(taps2, 1.0, white2)[256:256 + n_samples]

    low_weight = rng.uniform(0.25, 0.75)
    msg = low_weight * speech_low + (1.0 - low_weight) * speech

    for _ in range(int(rng.integers(0, 5))):
        f = rng.uniform(250.0, 3400.0)
        p = rng.uniform(0.0, 2.0 * np.pi)
        a = rng.uniform(0.02, 0.12)
        msg += a * np.sin(2.0 * np.pi * f * t + p)

    msg /= np.max(np.abs(msg)) + 1e-12
    msg *= rng.uniform(0.45, 1.0)

    mod_index = rng.uniform(0.20, 0.95)
    carrier_level = rng.uniform(0.55, 1.20)

    env = carrier_level + mod_index * msg
    env = np.maximum(env, 0.02)

    residual_offset = rng.uniform(-300.0, 300.0)
    phase = 2.0 * np.pi * residual_offset * t

    return (env * np.exp(1j * phase)).astype(np.complex64)


def generate_fm(n_samples):
    t = np.arange(n_samples) / FS
    f1 = rng.uniform(300.0, 1400.0)
    f2 = rng.uniform(1000.0, 3000.0)
    p1 = rng.uniform(0.0, 2.0 * np.pi)
    p2 = rng.uniform(0.0, 2.0 * np.pi)
    w1 = rng.uniform(0.4, 0.8)

    msg = (
        w1 * np.sin(2.0 * np.pi * f1 * t + p1)
        + (1.0 - w1) * np.sin(2.0 * np.pi * f2 * t + p2)
    )
    msg /= np.max(np.abs(msg)) + 1e-12

    deviation = rng.uniform(25_000.0, 75_000.0)
    phase = 2.0 * np.pi * deviation * np.cumsum(msg) / FS
    return np.exp(1j * phase).astype(np.complex64)


def generate_bpsk(n_samples):
    sps = int(round(FS / SYMBOL_RATE))
    n_symbols = int(np.ceil(n_samples / sps)) + 20
    bits = rng.integers(0, 2, n_symbols)
    symbols = (2.0 * bits - 1.0).astype(np.complex64)
    return shape_symbols(symbols, sps)[:n_samples]


def generate_qpsk(n_samples):
    sps = int(round(FS / SYMBOL_RATE))
    n_symbols = int(np.ceil(n_samples / sps)) + 20
    idx = rng.integers(0, 4, n_symbols)
    symbols = np.exp(1j * (np.pi / 4.0 + idx * np.pi / 2.0))
    return shape_symbols(symbols, sps)[:n_samples]


def generate_16qam(n_samples):
    sps = int(round(FS / SYMBOL_RATE))
    n_symbols = int(np.ceil(n_samples / sps)) + 20
    levels = np.array([-3, -1, 1, 3], dtype=np.float64)
    i = rng.choice(levels, n_symbols)
    q = rng.choice(levels, n_symbols)
    symbols = (i + 1j * q) / np.sqrt(10.0)
    return shape_symbols(symbols, sps)[:n_samples]


def generate_signal(label, n_samples):
    if label == "AM":
        return generate_am(n_samples)
    if label == "FM":
        return generate_fm(n_samples)
    if label == "BPSK":
        return generate_bpsk(n_samples)
    if label == "QPSK":
        return generate_qpsk(n_samples)
    if label == "16-QAM":
        return generate_16qam(n_samples)
    raise ValueError(label)


def apply_domain_randomization(iq):
    n = len(iq)
    t = np.arange(n) / FS

    static_offset = rng.uniform(-2000.0, 2000.0)
    jitter_amp = rng.uniform(0.0, 500.0)
    jitter_rate = rng.uniform(0.5, 40.0)
    jitter_phase = rng.uniform(0.0, 2.0 * np.pi)

    inst_freq = (
        static_offset
        + jitter_amp * np.sin(2.0 * np.pi * jitter_rate * t + jitter_phase)
    )
    phase_from_freq = 2.0 * np.pi * np.cumsum(inst_freq) / FS

    phase_noise_std = rng.uniform(0.002, 0.030)
    phase_noise = np.cumsum(rng.standard_normal(n) * phase_noise_std)

    out = iq * np.exp(1j * (phase_from_freq + phase_noise))

    fading_depth = rng.uniform(0.0, 0.12)
    fading_rate = rng.uniform(1.0, 50.0)
    fading_phase = rng.uniform(0.0, 2.0 * np.pi)
    fading = 1.0 + fading_depth * np.sin(
        2.0 * np.pi * fading_rate * t + fading_phase
    )

    gain = rng.uniform(0.70, 1.30)
    return (out * fading * gain).astype(np.complex64)


def preprocess(iq):
    filtered = signal.lfilter(FIR, 1.0, iq)
    filtered = filtered[200:]

    start = (len(filtered) - FINAL_SAMPLES) // 2
    segment = filtered[start:start + FINAL_SAMPLES]

    if len(segment) != FINAL_SAMPLES:
        raise ValueError("Konacni segment nema 5000 uzoraka.")

    power = np.mean(np.abs(segment) ** 2)
    if power > 1e-12:
        segment = segment / np.sqrt(power)

    return segment.astype(np.complex64)


print("=" * 90)
print("FINALNI AMC MODEL - 16 OBELEZJA")
print("=" * 90)
print("Klase:", CLASSES)
print("Obelezja:", len(FEATURE_COLUMNS))
print("Primeraka po klasi/SNR:", EXAMPLES_PER_CLASS_PER_SNR)

rows = []
for label in CLASSES:
    print("\nKLASA:", label)
    for snr_db in SNR_VALUES:
        print("  SNR", snr_db, "dB")
        for _ in range(EXAMPLES_PER_CLASS_PER_SNR):
            iq = generate_signal(label, RAW_SAMPLES)
            iq = apply_domain_randomization(iq)
            iq = add_awgn(iq, snr_db)
            iq = preprocess(iq)

            row = extract_features(iq, fs=FS)
            row["label"] = label
            row["snr_db"] = snr_db
            rows.append(row)


df = pd.DataFrame(rows)

nan_count = int(df[FEATURE_COLUMNS].isna().sum().sum())
finite_ok = bool(np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all())
if nan_count != 0 or not finite_ok:
    raise ValueError("Dataset sadrzi NaN ili beskonacne vrednosti.")

print("\nShape:", df.shape)
print("\nPo klasama:\n", df["label"].value_counts().sort_index())
df.to_csv(DATASET_NAME, index=False)

X = df[FEATURE_COLUMNS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    min_samples_split=2,
    random_state=SEED,
    n_jobs=-1,
)

print("\nTreniram Random Forest...")
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
report = classification_report(y_test, pred, digits=4)
cm = confusion_matrix(y_test, pred, labels=CLASSES)

print("\n" + "=" * 90)
print("SINTETICKI HOLDOUT")
print("=" * 90)
print(f"Accuracy: {acc * 100:.2f}%")
print(report)
print("Confusion matrix, klase:", CLASSES)
print(cm)

test_meta = df.loc[X_test.index, ["label", "snr_db"]].copy()
test_meta["prediction"] = pred

snr_results = []
print("\nTACNOST PO SNR-u")
for snr_db in SNR_VALUES:
    sub = test_meta[test_meta["snr_db"] == snr_db]
    sub_acc = np.mean(sub["label"].to_numpy() == sub["prediction"].to_numpy())
    snr_results.append((snr_db, sub_acc, len(sub)))
    print(f"SNR {snr_db:>3} dB: {sub_acc * 100:6.2f}% (n={len(sub)})")

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
importance_df.to_csv(IMPORTANCE_NAME, index=False)

print("\nFEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

with open(MODEL_NAME, "wb") as f:
    pickle.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "fs": FS,
        "final_samples": FINAL_SAMPLES,
        "classes": CLASSES,
        "snr_values": SNR_VALUES,
        "seed": SEED,
    }, f)

with open(RESULTS_NAME, "w", encoding="utf-8") as f:
    f.write("FINALNI AMC MODEL - 16 OBELEZJA\n")
    f.write("=" * 70 + "\n")
    f.write(f"Dataset: {DATASET_NAME}\n")
    f.write(f"Broj primera: {len(df)}\n")
    f.write(f"Accuracy: {acc * 100:.4f}%\n\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(str(cm))
    f.write("\n\nAccuracy by SNR:\n")
    for snr_db, sub_acc, n_sub in snr_results:
        f.write(f"{snr_db:>3} dB: {sub_acc * 100:.4f}% (n={n_sub})\n")
    f.write("\nFeature importance:\n")
    f.write(importance_df.to_string(index=False))

print("\nSacuvano:")
print("-", DATASET_NAME)
print("-", MODEL_NAME)
print("-", RESULTS_NAME)
print("-", IMPORTANCE_NAME)
print("\nSLEDECE: pokreni test_real_am_final.py bez ikakvog menjanja modela.")
