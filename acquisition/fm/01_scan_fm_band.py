import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from rtlsdr import RtlSdr
from scipy.signal import welch, find_peaks


# FM SKENER 87.5-108 MHz
# Cilj: pronaci vise jakih realnih FM stanica za finalni test

FM_START = 87.5e6
FM_END = 108.0e6

FS = 2.4e6
GAIN = 35.0

# Jedan prijem po centru
CAPTURE_SECONDS = 0.30
N_SAMPLES = int(FS * CAPTURE_SECONDS)

# Koristimo samo centralnih +/- 0.9 MHz svakog RTL-SDR zahvata
# da izbegnemo rubove spektra.
VALID_HALF_BW = 0.9e6

# Pomeraj centra pri skeniranju
CENTER_STEP = 1.6e6

# Ponovimo ceo opseg vise puta da ne zavisimo od jednog trenutka
NUM_PASSES = 3
PAUSE_BETWEEN_PASSES = 2.0

# Welch
NPERSEG = 4096
NOVERLAP = 2048

# Pik mora biti ovoliko iznad lokalne spektralne pozadine
DETECTION_THRESHOLD_DB = 8.0

# Minimalni razmak izmedju dva pika unutar jednog zahvata
MIN_PEAK_DISTANCE_HZ = 120e3

# Kasnije grupisemo detekcije koje su blize od ovoga
CLUSTER_TOLERANCE_HZ = 120e3

OUTPUT_DIR = "FM_SCAN_NEW_TEST_20260917"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_centers():
    first = FM_START + VALID_HALF_BW
    last = FM_END - VALID_HALF_BW

    centers = np.arange(
        first,
        last + 1,
        CENTER_STEP
    )

    # osiguraj da pokrijemo i gornji kraj
    if centers[-1] < last - 0.2e6:
        centers = np.append(
            centers,
            last
        )

    return centers


def analyze_capture(x, center_freq):
    x = x - np.mean(x)

    f, pxx = welch(
        x,
        fs=FS,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        return_onesided=False,
        scaling="density"
    )

    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)

    rf = center_freq + f
    psd_db = 10 * np.log10(pxx + 1e-20)

    mask = (
        (rf >= max(FM_START, center_freq - VALID_HALF_BW))
        &
        (rf <= min(FM_END, center_freq + VALID_HALF_BW))
    )

    rf_v = rf[mask]
    psd_v = psd_db[mask]

    noise_level = float(
        np.median(psd_v)
    )

    df = abs(
        rf_v[1] - rf_v[0]
    )

    min_distance_bins = max(
        1,
        int(MIN_PEAK_DISTANCE_HZ / df)
    )

    peaks, props = find_peaks(
        psd_v,
        height=noise_level + DETECTION_THRESHOLD_DB,
        distance=min_distance_bins
    )

    detections = []

    for peak_idx in peaks:
        peak_freq = float(
            rf_v[peak_idx]
        )

        peak_level = float(
            psd_v[peak_idx]
        )

        score = (
            peak_level -
            noise_level
        )

        detections.append({
            "frequency_hz": peak_freq,
            "frequency_mhz": peak_freq / 1e6,
            "score_db": score,
            "peak_level_db_hz": peak_level,
            "noise_level_db_hz": noise_level,
        })

    return detections


def cluster_detections(df):
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(
        "frequency_hz"
    ).reset_index(drop=True)

    clusters = []
    current = []

    for _, row in df.iterrows():
        if not current:
            current = [row]
            continue

        current_mean = np.mean(
            [r["frequency_hz"] for r in current]
        )

        if abs(
            row["frequency_hz"] - current_mean
        ) <= CLUSTER_TOLERANCE_HZ:
            current.append(row)
        else:
            clusters.append(current)
            current = [row]

    if current:
        clusters.append(current)

    summary = []

    for cluster in clusters:
        freqs = np.array(
            [r["frequency_hz"] for r in cluster],
            dtype=float
        )

        scores = np.array(
            [r["score_db"] for r in cluster],
            dtype=float
        )

        weights = np.maximum(
            scores,
            1e-6
        )

        representative_freq = np.average(
            freqs,
            weights=weights
        )

        summary.append({
            "frequency_mhz": representative_freq / 1e6,
            "max_score_db": float(np.max(scores)),
            "mean_score_db": float(np.mean(scores)),
            "detections": len(cluster),
        })

    result = pd.DataFrame(
        summary
    )

    return result.sort_values(
        ["max_score_db", "detections"],
        ascending=[False, False]
    ).reset_index(drop=True)


# ============================================================
# GLAVNI PROGRAM
# ============================================================

centers = make_centers()

print("=" * 90)
print("SKENIRANJE FM OPSEGA 87.5-108 MHz")
print("=" * 90)

print(
    f"Fs: {FS/1e6:.1f} MS/s"
)

print(
    f"Gain: {GAIN:.1f} dB"
)

print(
    f"Broj prolaza: {NUM_PASSES}"
)

print(
    f"Broj centralnih frekvencija po prolazu: {len(centers)}"
)

print(
    f"Trajanje prijema po centru: {CAPTURE_SECONDS:.2f} s"
)

print(
    f"Prag detekcije: {DETECTION_THRESHOLD_DB:.1f} dB"
)

print(
    f"Folder: {OUTPUT_DIR}"
)

print()

all_detections = []

sdr = RtlSdr()

try:
    sdr.sample_rate = FS
    sdr.gain = GAIN

    for pass_idx in range(
        1,
        NUM_PASSES + 1
    ):

        print()
        print(
            f"--- PROLAZ {pass_idx}/{NUM_PASSES} ---"
        )

        for center_freq in centers:
            sdr.center_freq = center_freq

            # kratko vreme da se tuner stabilizuje
            time.sleep(0.05)

            x = sdr.read_samples(
                N_SAMPLES
            )

            now = datetime.now()

            detections = analyze_capture(
                x,
                center_freq
            )

            print(
                f"{now.strftime('%H:%M:%S')} | "
                f"centar {center_freq/1e6:8.3f} MHz | "
                f"detekcije: {len(detections)}"
            )

            for d in detections:
                d["timestamp"] = str(now)
                d["pass"] = pass_idx
                d["center_freq_mhz"] = center_freq / 1e6

                all_detections.append(
                    d
                )

        if pass_idx < NUM_PASSES:
            time.sleep(
                PAUSE_BETWEEN_PASSES
            )

finally:
    sdr.close()


detections_df = pd.DataFrame(
    all_detections
)

raw_csv = os.path.join(
    OUTPUT_DIR,
    "fm_scan_all_detections.csv"
)

detections_df.to_csv(
    raw_csv,
    index=False
)

ranked_df = cluster_detections(
    detections_df
)

ranked_csv = os.path.join(
    OUTPUT_DIR,
    "fm_scan_ranked_stations.csv"
)

ranked_df.to_csv(
    ranked_csv,
    index=False
)

print()
print("=" * 90)
print("ZAVRSENO SKENIRANJE")
print("=" * 90)

print(
    f"Ukupan broj detekcija: {len(detections_df)}"
)

print()

if ranked_df.empty:
    print(
        "Nije pronadjena nijedna stanica iznad praga."
    )

else:
    print(
        "NAJJACI FM KANDIDATI:"
    )

    print(
        "-" * 90
    )

    top_n = min(
        15,
        len(ranked_df)
    )

    for i in range(
        top_n
    ):
        row = ranked_df.iloc[i]

        print(
            f"{i+1:2d}. "
            f"{row['frequency_mhz']:10.4f} MHz | "
            f"max={row['max_score_db']:6.2f} dB | "
            f"prosek={row['mean_score_db']:6.2f} dB | "
            f"detekcija={int(row['detections'])}"
        )

print()
print(
    f"Sirove detekcije: {raw_csv}"
)

print(
    f"Rangirane stanice: {ranked_csv}"
)

print(
    f"Rezultati sacuvani u: {OUTPUT_DIR}"
)
