# -*- coding: utf-8 -*-

import os
import time
import tempfile
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import features_final as feature_module


warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DejaVu Sans"



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
    / "efficiency"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RANDOM_STATE = 42

FS = 500_000.0
SYMBOL_RATE = 10_000.0

SNR_VALUES = [
    -10,
    -5,
    0,
    5,
    10,
    15,
    20
]

CLASSES = [
    "16-QAM",
    "AM",
    "BPSK",
    "FM",
    "QPSK"
]

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

SEGMENT_LENGTHS = [
    256,
    500,
    1000,
    2000,
    4000,
    5000,
    8000
]

# Broj ponavljanja kompletnog train/test eksperimenta
NUM_REPEATS_SEGMENT = 5

# Da eksperiment ne bude nepotrebno ogroman:
# 100 primera po klasi po SNR-u za svaku duzinu
EXAMPLES_PER_CLASS_PER_SNR_SEGMENT = 100


rng = np.random.default_rng(
    RANDOM_STATE
)


def normalize_power(x):

    power = np.mean(
        np.abs(x) ** 2
    )

    if power <= 0:
        return x

    return (
        x / np.sqrt(power)
    )


def add_awgn(
    x,
    snr_db
):

    x = normalize_power(
        x
    )

    signal_power = np.mean(
        np.abs(x) ** 2
    )

    snr_linear = (
        10.0 ** (
            snr_db / 10.0
        )
    )

    noise_power = (
        signal_power
        / snr_linear
    )

    noise = (
        rng.normal(
            0.0,
            np.sqrt(
                noise_power / 2.0
            ),
            len(x)
        )
        +
        1j
        * rng.normal(
            0.0,
            np.sqrt(
                noise_power / 2.0
            ),
            len(x)
        )
    )

    return (
        x + noise
    )


def add_impairments(
    x,
    fs
):

    phase_offset = rng.uniform(
        -np.pi,
        np.pi
    )

    freq_offset = rng.uniform(
        -1500.0,
        1500.0
    )

    n = np.arange(
        len(x)
    )

    rotator = np.exp(
        1j
        * (
            phase_offset
            +
            2.0
            * np.pi
            * freq_offset
            * n
            / fs
        )
    )

    return (
        x * rotator
    )

def generate_am(
    num_samples
):

    t = (
        np.arange(num_samples)
        / FS
    )

    modulation_frequency = rng.uniform(
        700.0,
        3500.0
    )

    modulation_index = rng.uniform(
        0.3,
        0.9
    )

    message = np.sin(
        2.0
        * np.pi
        * modulation_frequency
        * t
    )

    x = (
        1.0
        +
        modulation_index
        * message
    ).astype(
        np.complex128
    )

    return x


def generate_fm(
    num_samples
):

    t = (
        np.arange(num_samples)
        / FS
    )

    modulation_frequency = rng.uniform(
        500.0,
        4000.0
    )

    frequency_deviation = rng.uniform(
        20_000.0,
        75_000.0
    )

    message = np.sin(
        2.0
        * np.pi
        * modulation_frequency
        * t
    )

    phase = (
        2.0
        * np.pi
        * frequency_deviation
        * np.cumsum(message)
        / FS
    )

    return np.exp(
        1j * phase
    )


def generate_digital(
    modulation,
    num_samples
):

    samples_per_symbol = int(
        FS / SYMBOL_RATE
    )

    num_symbols = int(
        np.ceil(
            num_samples
            / samples_per_symbol
        )
    )


    if modulation == "BPSK":

        symbols = rng.choice(
            [
                -1.0,
                1.0
            ],
            size=num_symbols
        ).astype(
            np.complex128
        )


    elif modulation == "QPSK":

        constellation = np.array([
            1 + 1j,
            1 - 1j,
            -1 + 1j,
            -1 - 1j
        ]) / np.sqrt(2.0)

        symbols = rng.choice(
            constellation,
            size=num_symbols
        )


    elif modulation == "16-QAM":

        levels = np.array([
            -3,
            -1,
            1,
            3
        ])

        i_values = rng.choice(
            levels,
            size=num_symbols
        )

        q_values = rng.choice(
            levels,
            size=num_symbols
        )

        symbols = (
            i_values
            +
            1j * q_values
        ) / np.sqrt(10.0)


    else:

        raise ValueError(
            "Nepoznata digitalna modulacija."
        )


    x = np.repeat(
        symbols,
        samples_per_symbol
    )

    return x[
        :num_samples
    ]


def generate_signal(
    modulation,
    snr_db,
    num_samples
):

    if modulation == "AM":

        x = generate_am(
            num_samples
        )

    elif modulation == "FM":

        x = generate_fm(
            num_samples
        )

    else:

        x = generate_digital(
            modulation,
            num_samples
        )


    x = normalize_power(
        x
    )

    x = add_impairments(
        x,
        FS
    )

    x = add_awgn(
        x,
        snr_db
    )

    return x.astype(
        np.complex64
    )

def extract_features(
    iq
):

    result = feature_module.extract_features(
        iq,
        FS
    )

    values = np.array([
        result[name]
        for name in FEATURE_NAMES
    ],
        dtype=float
    )

    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


print()
print("=" * 100)
print("6.6.1 UTICAJ DUZINE I/Q SEGMENTA")
print("=" * 100)


segment_results = []


for num_samples in SEGMENT_LENGTHS:

    print()
    print("-" * 100)

    print(
        f"Duzina segmenta: "
        f"{num_samples} uzoraka"
    )


    X_list = []
    y_list = []


    total = (
        len(CLASSES)
        * len(SNR_VALUES)
        * EXAMPLES_PER_CLASS_PER_SNR_SEGMENT
    )

    counter = 0


    for snr_db in SNR_VALUES:

        for modulation in CLASSES:

            for _ in range(
                EXAMPLES_PER_CLASS_PER_SNR_SEGMENT
            ):

                iq = generate_signal(
                    modulation,
                    snr_db,
                    num_samples
                )

                features = extract_features(
                    iq
                )

                X_list.append(
                    features
                )

                y_list.append(
                    modulation
                )

                counter += 1


    X_seg = np.asarray(
        X_list,
        dtype=float
    )

    y_seg = np.asarray(
        y_list
    )


    accuracies = []


    for repeat in range(
        NUM_REPEATS_SEGMENT
    ):

        X_train, X_test, y_train, y_test = train_test_split(
            X_seg,
            y_seg,
            test_size=0.20,
            random_state=(
                RANDOM_STATE
                + repeat
            ),
            stratify=y_seg
        )


        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )


        rf.fit(
            X_train,
            y_train
        )

        prediction = rf.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        accuracies.append(
            accuracy
        )


    accuracies = np.asarray(
        accuracies
    )


    duration_ms = (
        num_samples
        / FS
        * 1000.0
    )

    approximate_symbols = (
        duration_ms
        / 1000.0
        * SYMBOL_RATE
    )


    mean_accuracy = (
        np.mean(accuracies)
        * 100.0
    )

    std_accuracy = (
        np.std(
            accuracies,
            ddof=1
        )
        * 100.0
    )


    segment_results.append({
        "num_samples": num_samples,
        "duration_ms": duration_ms,
        "approx_symbols": approximate_symbols,
        "mean_accuracy_percent": mean_accuracy,
        "std_accuracy_pp": std_accuracy
    })


    print(
        f"Trajanje: "
        f"{duration_ms:.3f} ms"
    )

    print(
        f"Priblizan broj simbola: "
        f"{approximate_symbols:.2f}"
    )

    print(
        f"Tacnost: "
        f"{mean_accuracy:.2f}% "
        f"± {std_accuracy:.2f} p.p."
    )


segment_df = pd.DataFrame(
    segment_results
)

segment_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "01_uticaj_duzine_segmenta.csv"
    ),
    index=False
)

plt.figure(
    figsize=(10, 6)
)

plt.errorbar(
    segment_df[
        "num_samples"
    ],
    segment_df[
        "mean_accuracy_percent"
    ],
    yerr=segment_df[
        "std_accuracy_pp"
    ],
    marker="o",
    linewidth=2,
    capsize=4
)

plt.axvline(
    5000,
    linestyle="--",
    linewidth=1.5,
    label="Изабрана дужина: 5000 узорака"
)

plt.xlabel(
    "Дужина I/Q сегмента [број узорака]"
)

plt.ylabel(
    "Тачност класификације [%]"
)

plt.title(
    "Утицај дужине I/Q сегмента на тачност класификације"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_uticaj_duzine_segmenta.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("=" * 100)
print("6.6.2 RACUNARSKA EFIKASNOST I MEMORIJSKI ZAHTEVI")
print("=" * 100)


df = pd.read_csv(
    DATA_FILE
)

X = df[
    FEATURE_NAMES
].values

y = df[
    "class"
].values


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

models = {

    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,

            # 1 za kontrolisano merenje
            n_jobs=1
        ),


    "SVM":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale"
                )
            )
        ]),


    "Decision Tree":

        DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE
        )
}


SINGLE_REPEATS = 500
BATCH_REPEATS = 20


efficiency_results = []


for model_name, model in models.items():

    print()
    print("-" * 100)

    print(
        "Model:",
        model_name
    )

    train_start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    train_time_s = (
        time.perf_counter()
        - train_start
    )

    test_prediction = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    one_example = X_test[
        0:1
    ]


    # warm-up
    for _ in range(10):

        model.predict(
            one_example
        )


    single_times = []


    for _ in range(
        SINGLE_REPEATS
    ):

        t0 = time.perf_counter()

        model.predict(
            one_example
        )

        t1 = time.perf_counter()

        single_times.append(
            (
                t1 - t0
            )
            * 1000.0
        )


    single_times = np.asarray(
        single_times
    )

    single_mean_ms = np.mean(
        single_times
    )

    single_std_ms = np.std(
        single_times,
        ddof=1
    )

    # warm-up
    model.predict(
        X_test
    )


    batch_times = []


    for _ in range(
        BATCH_REPEATS
    ):

        t0 = time.perf_counter()

        model.predict(
            X_test
        )

        t1 = time.perf_counter()

        batch_times.append(
            (
                t1 - t0
            )
            * 1000.0
        )


    batch_times = np.asarray(
        batch_times
    )

    batch_mean_ms = np.mean(
        batch_times
    )

    batch_std_ms = np.std(
        batch_times,
        ddof=1
    )

    batch_per_example_ms = (
        batch_mean_ms
        / len(X_test)
    )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".joblib",
        delete=False
    )

    temp_path = temp_file.name

    temp_file.close()


    joblib.dump(
        model,
        temp_path
    )

    model_size_kib = (
        os.path.getsize(
            temp_path
        )
        / 1024.0
    )

    os.remove(
        temp_path
    )


    efficiency_results.append({
        "model": model_name,
        "accuracy_percent": accuracy * 100.0,
        "train_time_s": train_time_s,
        "single_prediction_mean_ms": single_mean_ms,
        "single_prediction_std_ms": single_std_ms,
        "batch_test_mean_ms": batch_mean_ms,
        "batch_test_std_ms": batch_std_ms,
        "batch_per_example_ms": batch_per_example_ms,
        "model_size_kib": model_size_kib
    })


    print(
        f"Tacnost: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Obucavanje: "
        f"{train_time_s:.3f} s"
    )

    print(
        f"Jedna predikcija: "
        f"{single_mean_ms:.3f} "
        f"± {single_std_ms:.3f} ms"
    )

    print(
        f"Test skup: "
        f"{batch_mean_ms:.3f} "
        f"± {batch_std_ms:.3f} ms"
    )

    print(
        f"Velicina modela: "
        f"{model_size_kib:.2f} KiB"
    )


efficiency_df = pd.DataFrame(
    efficiency_results
)

efficiency_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "02_racunarska_efikasnost.csv"
    ),
    index=False
)


plt.figure(
    figsize=(8, 6)
)

x_pos = np.arange(
    len(efficiency_df)
)

plt.bar(
    x_pos,
    efficiency_df[
        "single_prediction_mean_ms"
    ],
    yerr=efficiency_df[
        "single_prediction_std_ms"
    ],
    capsize=5
)

plt.xticks(
    x_pos,
    [
        "Random Forest",
        "SVM",
        "Decision Tree"
    ]
)

plt.ylabel(
    "Време једне предикције [ms]"
)

plt.xlabel(
    "Класификациони модел"
)

plt.title(
    "Поређење времена предикције једног примера"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_vreme_jedne_predikcije.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(
    figsize=(8, 6)
)

plt.bar(
    x_pos,
    efficiency_df[
        "model_size_kib"
    ]
)

plt.xticks(
    x_pos,
    [
        "Random Forest",
        "SVM",
        "Decision Tree"
    ]
)

plt.ylabel(
    "Величина модела [KiB]"
)

plt.xlabel(
    "Класификациони модел"
)

plt.title(
    "Поређење меморијских захтева класификационих модела"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_velicina_modela.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("=" * 100)
print("6.6.1 REZULTATI DUZINE SEGMENTA")
print("=" * 100)

print(
    segment_df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("6.6.2 RACUNARSKA EFIKASNOST")
print("=" * 100)

print(
    efficiency_df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("SACUVANO U")
print("=" * 100)

print(
    OUTPUT_DIR
)

print()
print("GOTOVO.")