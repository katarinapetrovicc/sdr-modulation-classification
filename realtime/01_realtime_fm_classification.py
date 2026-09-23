# -*- coding: utf-8 -*-

import os
import pickle
import queue
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rtlsdr import RtlSdr
from scipy import signal

from features_final import extract_features, FEATURE_COLUMNS


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "amc_model_final_16features_500ksps_10ms.pkl"
)

OUTPUT_DIR = os.path.join( PROJECT_ROOT, "REALTIME_106_3MHz_10MIN_RESULTS" )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUTPUT_DIR, "realtime_10min_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "realtime_10min_summary.txt")
COMPONENTS_FIG = os.path.join(OUTPUT_DIR, "01_prosecna_vremena_faza.png")
LATENCY_FIG = os.path.join(OUTPUT_DIR, "02_vreme_obrade_blokova.png")
QUEUE_FIG = os.path.join(OUTPUT_DIR, "03_popunjenost_reda.png")
CLASSES_FIG = os.path.join(OUTPUT_DIR, "04_raspodela_klasa.png")

FS_SDR = 2_400_000
FS_ML = 500_000

STATION_FREQ = 98.7e6
CENTER_FREQ = 99.0e6
FREQ_OFFSET = STATION_FREQ - CENTER_FREQ

BLOCK_DURATION_S = 0.050
BLOCK_SAMPLES = int(FS_SDR * BLOCK_DURATION_S)

ML_SEGMENT_DURATION_S = 0.010
ML_SEGMENT_SAMPLES = int(FS_ML * ML_SEGMENT_DURATION_S)

# 12000 * 50 ms = 600 s = 10 min posmatranog signala
NUM_BLOCKS = 12_000

QUEUE_MAXSIZE = 5

FILTER_NUM_TAPS = 129
FILTER_CUTOFF_HZ = 100_000
TRANSIENT_OUTPUT_SAMPLES = 200

DISCARD_SAMPLES = 65_536
REALTIME_LIMIT_MS = BLOCK_DURATION_S * 1000.0
PROGRESS_EVERY = 100


if BLOCK_SAMPLES != 120_000:
    raise ValueError("Blok mora imati 120000 uzoraka.")

if ML_SEGMENT_SAMPLES != 5_000:
    raise ValueError("ML segment mora imati 5000 uzoraka.")


print("=" * 100)
print("DESETOMINUTNI REAL-TIME SDR TEST - 106.3 MHz")
print("=" * 100)

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "Model nije pronadjen:\n{}".format(MODEL_FILE)
    )

with open(MODEL_FILE, "rb") as model_handle:
    bundle = pickle.load(model_handle)

model = bundle["model"]
MODEL_FEATURE_COLUMNS = list(bundle["feature_columns"])
MODEL_FS = int(bundle["fs"])
MODEL_SAMPLES = int(bundle["final_samples"])

if MODEL_FEATURE_COLUMNS != FEATURE_COLUMNS:
    raise ValueError(
        "FEATURE_COLUMNS modela i features_final.py se ne poklapaju."
    )

if MODEL_FS != FS_ML:
    raise ValueError(
        "Model ocekuje fs={}, a pipeline daje {}.".format(
            MODEL_FS,
            FS_ML
        )
    )

if MODEL_SAMPLES != ML_SEGMENT_SAMPLES:
    raise ValueError(
        "Model ocekuje {} uzoraka, a pipeline daje {}.".format(
            MODEL_SAMPLES,
            ML_SEGMENT_SAMPLES
        )
    )

if hasattr(model, "n_jobs"):
    model.set_params(n_jobs=1)

MODEL_CLASSES = [str(class_name) for class_name in model.classes_]

print("Model:", MODEL_FILE)
print("Broj obelezja:", len(MODEL_FEATURE_COLUMNS))
print("Klase:", MODEL_CLASSES)


fir_coeff = signal.firwin(
    FILTER_NUM_TAPS,
    FILTER_CUTOFF_HZ,
    fs=FS_SDR
)

sample_times = np.arange(BLOCK_SAMPLES) / FS_SDR

# Signal je na -300 kHz u odnosu na centar prijemnika
# Ovim mnozenjem pomera se na 0 Hz
mixer = np.exp(
    -1j * 2.0 * np.pi * FREQ_OFFSET * sample_times
)


def probability_column_name(class_name):
    """Formira citljivo ime CSV kolone za verovatnocu klase."""

    safe_name = str(class_name).replace("-", "_").replace(" ", "_")
    return "probability_{}".format(safe_name)

def process_block(samples):

    if len(samples) != BLOCK_SAMPLES:
        raise ValueError(
            "Ocekivano {} uzoraka, dobijeno {}.".format(
                BLOCK_SAMPLES,
                len(samples)
            )
        )

    wall_start = time.perf_counter()

    # DSP
    start = time.perf_counter()

    shifted = samples * mixer
    filtered = signal.lfilter(fir_coeff, 1.0, shifted)
    resampled = signal.resample_poly(filtered, up=5, down=24)

    if len(resampled) <= TRANSIENT_OUTPUT_SAMPLES:
        raise ValueError("Premalo uzoraka nakon resampliranja.")

    resampled = resampled[TRANSIENT_OUTPUT_SAMPLES:]

    if len(resampled) < ML_SEGMENT_SAMPLES:
        raise ValueError("Nema dovoljno uzoraka za ML segment od 10 ms.")

    segment_start = (len(resampled) - ML_SEGMENT_SAMPLES) // 2

    iq_segment = resampled[
        segment_start:segment_start + ML_SEGMENT_SAMPLES
    ]

    power = np.mean(np.abs(iq_segment) ** 2)

    if power > 1e-12:
        iq_segment = iq_segment / np.sqrt(power)

    dsp_ms = (time.perf_counter() - start) * 1000.0

    # Izdvajanje obelezja
    start = time.perf_counter()

    features = extract_features(iq_segment, fs=FS_ML)

    feature_values = np.asarray(
        [features[name] for name in MODEL_FEATURE_COLUMNS],
        dtype=np.float64
    )

    if not np.all(np.isfinite(feature_values)):
        raise ValueError("Dobijena su nevalidna obelezja.")

    feature_ms = (time.perf_counter() - start) * 1000.0

    model_input = pd.DataFrame(
        [feature_values],
        columns=MODEL_FEATURE_COLUMNS
    )

    # ML predikcija
    start = time.perf_counter()

    probabilities = model.predict_proba(model_input)[0]

    ml_ms = (time.perf_counter() - start) * 1000.0

    best_index = int(np.argmax(probabilities))
    prediction = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    sorted_probabilities = np.sort(probabilities)
    margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])

    phase_sum_ms = dsp_ms + feature_ms + ml_ms
    wall_processing_ms = (time.perf_counter() - wall_start) * 1000.0

    result = {
        "prediction": prediction,
        "confidence": confidence,
        "margin": margin,
        "dsp_ms": dsp_ms,
        "feature_ms": feature_ms,
        "ml_ms": ml_ms,
        "phase_sum_ms": phase_sum_ms,
        "wall_processing_ms": wall_processing_ms,
        "below_50ms": wall_processing_ms < REALTIME_LIMIT_MS
    }

    for class_name, probability in zip(model.classes_, probabilities):
        result[probability_column_name(class_name)] = float(probability)

    return result


print()
print("Warm-up DSP + features + ML lanca...")

rng = np.random.default_rng(42)

dummy = (
    rng.standard_normal(BLOCK_SAMPLES)
    + 1j * rng.standard_normal(BLOCK_SAMPLES)
).astype(np.complex64)

for _ in range(3):
    process_block(dummy)

print("Warm-up zavrsen.")


print()
print("=" * 100)
print("PARAMETRI SISTEMA")
print("=" * 100)
print("Center: {:.3f} MHz".format(CENTER_FREQ / 1e6))
print("FM signal: {:.3f} MHz".format(STATION_FREQ / 1e6))
print("RTL-SDR fs: {:.1f} MS/s".format(FS_SDR / 1e6))
print("ML fs: {:.0f} kS/s".format(FS_ML / 1e3))
print("Blok: {:.1f} ms".format(BLOCK_DURATION_S * 1000.0))
print("Uzoraka po bloku:", BLOCK_SAMPLES)
print("ML segment: {} uzoraka".format(ML_SEGMENT_SAMPLES))
print("Broj blokova:", NUM_BLOCKS)
print("Trajanje posmatranog signala: {:.1f} s".format(
    NUM_BLOCKS * BLOCK_DURATION_S
))
print("Kapacitet reda:", QUEUE_MAXSIZE)


block_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

results = []
queue_sizes = []
errors = []

received_blocks = 0
processed_blocks = 0
dropped_blocks = 0
processing_error_blocks = 0

lock = threading.Lock()


def acquisition_worker(sdr):

    global received_blocks
    global dropped_blocks

    try:
        for block_index in range(NUM_BLOCKS):

            acquisition_start = time.perf_counter()
            samples = sdr.read_samples(BLOCK_SAMPLES)
            acquisition_ms = (
                time.perf_counter() - acquisition_start
            ) * 1000.0

            data = {
                "block": block_index + 1,
                "samples": np.asarray(samples, dtype=np.complex64),
                "acquisition_ms": acquisition_ms,
                "arrival_time": time.perf_counter()
            }

            with lock:
                received_blocks += 1

            try:
                block_queue.put_nowait(data)
                queue_sizes.append(block_queue.qsize())

            except queue.Full:
                with lock:
                    dropped_blocks += 1

                print(
                    "Blok {} je odbacen jer je red pun.".format(
                        block_index + 1
                    )
                )

    except Exception as exc:
        errors.append("RTL-SDR prijem: {}".format(exc))

    finally:
        # Blokira samo ako je red trenutno pun, dok consumer ne oslobodi mesto.
        block_queue.put(None)



def processing_worker():

    global processed_blocks
    global processing_error_blocks

    while True:
        data = block_queue.get()

        try:
            if data is None:
                break

            queue_wait_ms = (
                time.perf_counter() - data["arrival_time"]
            ) * 1000.0

            result = process_block(data["samples"])

            result.update({
                "block": data["block"],
                "acquisition_ms": data["acquisition_ms"],
                "queue_wait_ms": queue_wait_ms,
                "queue_size_after_get": block_queue.qsize()
            })

            results.append(result)

            with lock:
                processed_blocks += 1

            if data["block"] == 1 or data["block"] % PROGRESS_EVERY == 0:
                print(
                    "Blok {}/{} | klasa: {} | obrada: {:.3f} ms | red: {}/{}".format(
                        data["block"],
                        NUM_BLOCKS,
                        result["prediction"],
                        result["wall_processing_ms"],
                        result["queue_size_after_get"],
                        QUEUE_MAXSIZE
                    )
                )

        except Exception as exc:
            with lock:
                processing_error_blocks += 1

            block_number = "nepoznat"
            if data is not None:
                block_number = data.get("block", "nepoznat")

            errors.append(
                "Obrada bloka {}: {}".format(block_number, exc)
            )

        finally:
            block_queue.task_done()


print()
print("=" * 100)
print("POCETAK DESETOMINUTNOG REAL-TIME TESTA")
print("=" * 100)

sdr = RtlSdr()
sdr.sample_rate = FS_SDR
sdr.center_freq = CENTER_FREQ
sdr.gain = "auto"

print("Stabilizacija prijemnika...")
_ = sdr.read_samples(DISCARD_SAMPLES)
print("Stabilizacija zavrsena.")

producer = threading.Thread(
    target=acquisition_worker,
    args=(sdr,),
    name="rtl_sdr_acquisition"
)

consumer = threading.Thread(
    target=processing_worker,
    name="dsp_ml_processing"
)

test_start = time.perf_counter()

try:
    consumer.start()
    producer.start()

    producer.join()
    block_queue.join()
    consumer.join()

finally:
    sdr.close()

test_duration_s = time.perf_counter() - test_start


results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("Nijedan blok nije uspesno obradjen.")

results_df = results_df.sort_values("block").reset_index(drop=True)
results_df.to_csv(CSV_FILE, index=False)


latency = results_df["wall_processing_ms"]

mean_acquisition = results_df["acquisition_ms"].mean()
mean_dsp = results_df["dsp_ms"].mean()
mean_features = results_df["feature_ms"].mean()
mean_ml = results_df["ml_ms"].mean()
mean_phase_sum = results_df["phase_sum_ms"].mean()

mean_processing = latency.mean()
median_processing = latency.median()
p95_processing = latency.quantile(0.95)
p99_processing = latency.quantile(0.99)
max_processing = latency.max()

deadline_misses = int((latency >= REALTIME_LIMIT_MS).sum())
percentage_below_50 = float((latency < REALTIME_LIMIT_MS).mean() * 100.0)

max_queue = max(queue_sizes) if queue_sizes else 0
mean_queue = float(np.mean(queue_sizes)) if queue_sizes else 0.0

class_counts = results_df["prediction"].value_counts().sort_index()
class_percentages = (
    results_df["prediction"].value_counts(normalize=True).sort_index() * 100.0
)

mean_confidence = results_df["confidence"].mean()
mean_margin = results_df["margin"].mean()

fm_probability_column = probability_column_name("FM")

if fm_probability_column in results_df.columns:
    mean_fm_probability = results_df[fm_probability_column].mean()
else:
    mean_fm_probability = np.nan

fm_count = int((results_df["prediction"] == "FM").sum())
fm_percentage = 100.0 * fm_count / len(results_df)

print()
print("=" * 100)
print("FINALNI REZULTATI")
print("=" * 100)
print("Primljeno blokova:", received_blocks)
print("Obradjeno blokova:", processed_blocks)
print("Odbaceno zbog punog reda:", dropped_blocks)
print("Greske pri obradi:", processing_error_blocks)
print("Stvarno trajanje testa: {:.3f} s".format(test_duration_s))
print("Trajanje obuhvacenog signala: {:.3f} s".format(
    received_blocks * BLOCK_DURATION_S
))

print()
print("VREMENA")
print("Prosecno vreme prijema: {:.3f} ms".format(mean_acquisition))
print("Prosecno DSP vreme: {:.3f} ms".format(mean_dsp))
print("Prosecno vreme obelezja: {:.3f} ms".format(mean_features))
print("Prosecno ML vreme: {:.3f} ms".format(mean_ml))
print("Prosecan zbir faza: {:.3f} ms".format(mean_phase_sum))
print("Prosecno ukupno vreme obrade: {:.3f} ms".format(mean_processing))
print("Medijana ukupnog vremena obrade: {:.3f} ms".format(median_processing))
print("95. percentil: {:.3f} ms".format(p95_processing))
print("99. percentil: {:.3f} ms".format(p99_processing))
print("Maksimalno vreme obrade: {:.3f} ms".format(max_processing))
print("Blokovi sa obradom >= 50 ms:", deadline_misses)
print("Blokovi sa obradom < 50 ms: {:.3f}%".format(percentage_below_50))

print()
print("RED")
print("Maksimalna popunjenost: {}/{}".format(max_queue, QUEUE_MAXSIZE))
print("Prosecna popunjenost: {:.3f}".format(mean_queue))

print()
print("KLASIFIKACIJA")
print("FM predikcije: {}/{} ({:.3f}%)".format(
    fm_count,
    len(results_df),
    fm_percentage
))
print("Prosecna verovatnoca FM klase: {:.6f}".format(mean_fm_probability))
print("Prosecna pouzdanost pobednicke klase: {:.6f}".format(mean_confidence))
print("Prosecna margina: {:.6f}".format(mean_margin))

for class_name in class_counts.index:
    print(
        "{}: {} ({:.3f}%)".format(
            class_name,
            int(class_counts[class_name]),
            float(class_percentages[class_name])
        )
    )

if errors:
    print()
    print("GRESKE")
    for error in errors:
        print("-", error)


with open(SUMMARY_FILE, "w", encoding="utf-8") as summary:

    summary.write("DESETOMINUTNI REAL-TIME SDR TEST - 106.3 MHz\n")
    summary.write("=" * 75 + "\n\n")

    summary.write("Centar prijemnika: {:.3f} MHz\n".format(
        CENTER_FREQ / 1e6
    ))
    summary.write("Posmatrani signal: {:.3f} MHz\n".format(
        STATION_FREQ / 1e6
    ))
    summary.write("Frekvencija uzorkovanja: {:.1f} MS/s\n".format(
        FS_SDR / 1e6
    ))
    summary.write("Trajanje bloka: {:.1f} ms\n".format(
        BLOCK_DURATION_S * 1000.0
    ))
    summary.write("Kapacitet reda: {}\n\n".format(QUEUE_MAXSIZE))

    summary.write("Primljeno blokova: {}\n".format(received_blocks))
    summary.write("Obradjeno blokova: {}\n".format(processed_blocks))
    summary.write("Odbaceno zbog punog reda: {}\n".format(dropped_blocks))
    summary.write("Greske pri obradi: {}\n".format(processing_error_blocks))
    summary.write("Stvarno trajanje testa: {:.3f} s\n".format(
        test_duration_s
    ))
    summary.write("Trajanje obuhvacenog signala: {:.3f} s\n\n".format(
        received_blocks * BLOCK_DURATION_S
    ))

    summary.write("Prosecno vreme prijema: {:.3f} ms\n".format(
        mean_acquisition
    ))
    summary.write("Prosecno DSP vreme: {:.3f} ms\n".format(mean_dsp))
    summary.write("Prosecno vreme obelezja: {:.3f} ms\n".format(
        mean_features
    ))
    summary.write("Prosecno ML vreme: {:.3f} ms\n".format(mean_ml))
    summary.write("Prosecan zbir faza: {:.3f} ms\n".format(mean_phase_sum))
    summary.write("Prosecno ukupno vreme obrade: {:.3f} ms\n".format(
        mean_processing
    ))
    summary.write("Medijana ukupnog vremena obrade: {:.3f} ms\n".format(
        median_processing
    ))
    summary.write("95. percentil: {:.3f} ms\n".format(p95_processing))
    summary.write("99. percentil: {:.3f} ms\n".format(p99_processing))
    summary.write("Maksimalno vreme obrade: {:.3f} ms\n".format(
        max_processing
    ))
    summary.write("Blokovi sa obradom >= 50 ms: {}\n".format(
        deadline_misses
    ))
    summary.write("Blokovi sa obradom < 50 ms: {:.3f}%\n\n".format(
        percentage_below_50
    ))

    summary.write("Maksimalna popunjenost reda: {}/{}\n".format(
        max_queue,
        QUEUE_MAXSIZE
    ))
    summary.write("Prosecna popunjenost reda: {:.3f}\n\n".format(
        mean_queue
    ))

    summary.write("FM predikcije: {}/{} ({:.3f}%)\n".format(
        fm_count,
        len(results_df),
        fm_percentage
    ))
    summary.write("Prosecna verovatnoca FM klase: {:.6f}\n".format(
        mean_fm_probability
    ))
    summary.write("Prosecna pouzdanost pobednicke klase: {:.6f}\n".format(
        mean_confidence
    ))
    summary.write("Prosecna margina: {:.6f}\n\n".format(mean_margin))

    summary.write("Raspodela predvidjenih klasa:\n")

    for class_name in class_counts.index:
        summary.write(
            "{}: {} ({:.3f}%)\n".format(
                class_name,
                int(class_counts[class_name]),
                float(class_percentages[class_name])
            )
        )

    if errors:
        summary.write("\nGreske:\n")
        for error in errors:
            summary.write("- {}\n".format(error))


plt.rcParams["font.family"] = "DejaVu Sans"

# Slika 1 - prosecna vremena faza
phase_names = [
    "DSP обрада",
    "Издвајање\nобележја",
    "ML предикција"
]

phase_values = [mean_dsp, mean_features, mean_ml]

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(phase_names, phase_values)
ax.set_ylabel("Просечно време [ms]")
ax.set_title("Просечно време извршавања фаза обраде")
ax.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, phase_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        value,
        "{:.3f}".format(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(COMPONENTS_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)

# Slika 2 - vreme obrade svih blokova
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(
    results_df["block"],
    results_df["wall_processing_ms"],
    linewidth=0.7,
    label="Време обраде"
)
ax.axhline(
    REALTIME_LIMIT_MS,
    color="red",
    linestyle="--",
    label="Граница 50 ms"
)
ax.axhline(
    p99_processing,
    color="orange",
    linestyle=":",
    label="99. перцентил"
)
ax.set_xlabel("Редни број блока")
ax.set_ylabel("Време обраде [ms]")
ax.set_title("Време обраде појединачних блокова")
ax.set_xticks(np.linspace(1, NUM_BLOCKS, 11, dtype=int))
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(LATENCY_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)

# Slika 3 - popunjenost reda
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.step(
    np.arange(1, len(queue_sizes) + 1),
    queue_sizes,
    where="mid",
    linewidth=0.7
)
ax.axhline(
    QUEUE_MAXSIZE,
    color="red",
    linestyle="--",
    label="Капацитет реда"
)
ax.set_xlabel("Редни број примљеног блока")
ax.set_ylabel("Број блокова у реду")
ax.set_title("Попуњеност реда током експеримента")
ax.set_xticks(np.linspace(1, len(queue_sizes), 11, dtype=int))
ax.set_yticks(range(0, QUEUE_MAXSIZE + 1))
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(QUEUE_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)

# Slika 4 - raspodela predvidjenih klasa
fig, ax = plt.subplots(figsize=(8, 5.5))
class_bars = ax.bar(class_counts.index.astype(str), class_counts.values)
ax.set_xlabel("Предвиђена класа")
ax.set_ylabel("Број блокова")
ax.set_title("Расподела предвиђених класа")
ax.grid(axis="y", alpha=0.3)

for bar, value in zip(class_bars, class_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        value,
        str(int(value)),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(CLASSES_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)



print()
print("=" * 100)
print("SACUVANO")
print("=" * 100)
print(CSV_FILE)
print(SUMMARY_FILE)
print(COMPONENTS_FIG)
print(LATENCY_FIG)
print(QUEUE_FIG)
print(CLASSES_FIG)

