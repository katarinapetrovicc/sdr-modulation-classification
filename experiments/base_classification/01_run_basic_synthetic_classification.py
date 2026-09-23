# -*- coding: utf-8 -*-

import os
import time
import warnings
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.inspection import permutation_importance


import features_final as feature_module


warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DejaVu Sans"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "base_classification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



RANDOM_STATE = 42

FS = 500_000.0

DURATION = 0.010

NUM_SAMPLES = int(
    FS * DURATION
)

SYMBOL_RATE = 10_000.0

SAMPLES_PER_SYMBOL = int(
    FS / SYMBOL_RATE
)

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

EXAMPLES_PER_CLASS_PER_SNR = 250


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


CLASS_LABELS_CYR = {
    "16-QAM": "16-QAM",
    "AM": "AM",
    "BPSK": "BPSK",
    "FM": "FM",
    "QPSK": "QPSK"
}


MODEL_LABELS_CYR = {
    "Logistic Regression": "Логистичка регресија",
    "Gaussian Naive Bayes": "Gaussian Naive Bayes",
    "KNN": "KNN",
    "Random Forest": "Random Forest",
    "Decision Tree": "Стабло одлучивања",
    "SVM": "SVM"
}


FEATURE_LABELS_CYR = {
    "amp_var": "Варијанса амплитуде",
    "amp_kurtosis": "Куртоза амплитуде",
    "amp_skew": "Асиметрија амплитуде",
    "phase_var": "Варијанса фазе",
    "phase_kurtosis": "Куртоза фазе",
    "freq_var": "Варијанса тренутне фреквенције",
    "freq_kurtosis": "Куртоза тренутне фреквенције",
    "center_power": "Снага централног дела спектра",
    "spectral_flatness": "Спектрална равност",
    "C20_abs": "|C20|",
    "C21_abs": "|C21|",
    "C40_abs": "|C40|",
    "C42_abs": "|C42|",
    "carrier_power_ratio": "Однос снаге носиоца",
    "sideband_symmetry": "Симетрија бочних опсега",
    "envelope_modulation_index": "Индекс модулације омотача"
}

rng = np.random.default_rng(
    RANDOM_STATE
)


def normalize_power(x):

    p = np.mean(
        np.abs(x) ** 2
    )

    if p <= 0:
        return x

    return (
        x
        / np.sqrt(p)
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
        x
        + noise
    )


def random_impairments(
    x
):

    # nasumicna pocetna faza
    phase_offset = rng.uniform(
        -np.pi,
        np.pi
    )

    # mali frekvencijski pomeraj
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
            / FS
        )
    )

    return (
        x
        * rotator
    )

def generate_am():

    t = (
        np.arange(NUM_SAMPLES)
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

    modulating = np.sin(
        2.0
        * np.pi
        * modulation_frequency
        * t
    )

    x = (
        1.0
        +
        modulation_index
        * modulating
    ).astype(
        np.complex128
    )

    return x


def generate_fm():

    t = (
        np.arange(NUM_SAMPLES)
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

    modulating = np.sin(
        2.0
        * np.pi
        * modulation_frequency
        * t
    )

    phase = (
        2.0
        * np.pi
        * frequency_deviation
        * np.cumsum(
            modulating
        )
        / FS
    )

    x = np.exp(
        1j
        * phase
    )

    return x


def generate_symbols(
    modulation
):

    num_symbols = int(
        np.ceil(
            NUM_SAMPLES
            / SAMPLES_PER_SYMBOL
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

        i_part = rng.choice(
            levels,
            size=num_symbols
        )

        q_part = rng.choice(
            levels,
            size=num_symbols
        )

        symbols = (
            i_part
            +
            1j
            * q_part
        ) / np.sqrt(10.0)


    else:

        raise ValueError(
            "Nepoznata digitalna modulacija."
        )


    x = np.repeat(
        symbols,
        SAMPLES_PER_SYMBOL
    )

    x = x[
        :NUM_SAMPLES
    ]

    return x


def generate_signal(
    modulation,
    snr_db
):

    if modulation == "AM":

        x = generate_am()


    elif modulation == "FM":

        x = generate_fm()


    elif modulation in [
        "BPSK",
        "QPSK",
        "16-QAM"
    ]:

        x = generate_symbols(
            modulation
        )


    else:

        raise ValueError(
            f"Nepoznata klasa: {modulation}"
        )


    x = normalize_power(
        x
    )

    x = random_impairments(
        x
    )

    x = add_awgn(
        x,
        snr_db
    )

    return x.astype(
        np.complex64
    )

def find_feature_function():

    possible_names = [
        "extract_features",
        "extract_features_final",
        "compute_features",
        "get_features",
        "features"
    ]

    for name in possible_names:

        if hasattr(
            feature_module,
            name
        ):

            fn = getattr(
                feature_module,
                name
            )

            if callable(fn):

                print(
                    "Koristi se funkcija za obelezja:",
                    name
                )

                return fn


    # ako naziv nije medju ocekivanim,
    # probaj da pronadjes javnu funkciju
    candidates = []

    for name in dir(
        feature_module
    ):

        if name.startswith("_"):
            continue

        obj = getattr(
            feature_module,
            name
        )

        if callable(obj):

            candidates.append(
                name
            )


    raise RuntimeError(
        "Nije pronadjena funkcija za izdvajanje obelezja u "
        "features_final_multirate.py.\n"
        "Pronadjene funkcije su: "
        + ", ".join(candidates)
    )


FEATURE_FUNCTION = find_feature_function()


def extract_final_features(
    x
):

    """
    Pokusava poziv finalne funkcije sa:
    (x, fs) ili samo (x).
    """

    try:

        result = FEATURE_FUNCTION(
            x,
            FS
        )

    except TypeError:

        result = FEATURE_FUNCTION(
            x
        )


    # ako funkcija vraca dict
    if isinstance(
        result,
        dict
    ):

        values = np.array([
            result[name]
            for name in FEATURE_NAMES
        ],
            dtype=float
        )


    else:

        values = np.asarray(
            result,
            dtype=float
        ).ravel()


    if len(values) != 16:

        raise ValueError(
            "Finalna funkcija za obelezja mora da vrati 16 vrednosti, "
            f"a vraceno je {len(values)}."
        )


    if not np.all(
        np.isfinite(values)
    ):

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


    return values

print()
print("=" * 100)
print("GENERISANJE OSNOVNOG PETOKLASNOG SINTETICKOG SKUPA")
print("=" * 100)

X_list = []
y_list = []
snr_list = []

total_examples = (
    len(CLASSES)
    * len(SNR_VALUES)
    * EXAMPLES_PER_CLASS_PER_SNR
)

counter = 0


for snr_db in SNR_VALUES:

    for modulation in CLASSES:

        print(
            f"SNR = {snr_db:>3} dB, "
            f"klasa = {modulation}"
        )

        for _ in range(
            EXAMPLES_PER_CLASS_PER_SNR
        ):

            iq = generate_signal(
                modulation,
                snr_db
            )

            features = extract_final_features(
                iq
            )

            X_list.append(
                features
            )

            y_list.append(
                modulation
            )

            snr_list.append(
                snr_db
            )

            counter += 1


            if counter % 500 == 0:

                print(
                    f"Obradjeno "
                    f"{counter}/{total_examples}"
                )


X = np.asarray(
    X_list,
    dtype=float
)

y = np.asarray(
    y_list
)

snr_array = np.asarray(
    snr_list
)


print()
print(
    "Dimenzija skupa:",
    X.shape
)

print(
    "Ukupan broj primera:",
    len(y)
)


dataset_df = pd.DataFrame(
    X,
    columns=FEATURE_NAMES
)

dataset_df[
    "class"
] = y

dataset_df[
    "snr_db"
] = snr_array

DATASET_CSV = os.path.join(
    OUTPUT_DIR,
    "00_osnovni_sinteticki_skup_16_obelezja.csv"
)

dataset_df.to_csv(
    DATASET_CSV,
    index=False
)


X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X,
    y,
    snr_array,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print()
print("=" * 100)
print("GLAVNA PODELA SKUPA")
print("=" * 100)

print(
    "Trening primera:",
    len(y_train)
)

print(
    "Test primera:",
    len(y_test)
)


models = {

    "Logistic Regression":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE
                )
            )
        ]),


    "Gaussian Naive Bayes":

        GaussianNB(),


    "KNN":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=7,
                    weights="distance"
                )
            )
        ]),


    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),


    "Decision Tree":

        DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE
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
        ])
}


print()
print("=" * 100)
print("6.5.1 POREDJENJE KLASIFIKACIONIH ALGORITAMA")
print("=" * 100)


comparison_results = []

trained_models = {}


for model_name, model in models.items():

    print()
    print(
        "Model:",
        model_name
    )


    start_train = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    train_time = (
        time.perf_counter()
        - start_train
    )


    start_prediction = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter()
        - start_prediction
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )


    comparison_results.append({
        "algorithm": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "train_time_s": train_time,
        "test_prediction_time_ms": prediction_time * 1000.0
    })


    trained_models[
        model_name
    ] = model


    print(
        f"TACNOST: "
        f"{accuracy * 100:.2f}%"
    )


comparison_df = pd.DataFrame(
    comparison_results
)

comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "01_poredjenje_algoritama.csv"
    ),
    index=False
)

plt.figure(
    figsize=(10, 6)
)

x_pos = np.arange(
    len(comparison_df)
)

values = (
    comparison_df[
        "accuracy"
    ].values
    * 100.0
)

bars = plt.bar(
    x_pos,
    values
)

plt.xticks(
    x_pos,
    [
        MODEL_LABELS_CYR[
            name
        ]
        for name in comparison_df[
            "algorithm"
        ]
    ],
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Тачност [%]"
)

plt.xlabel(
    "Класификациони алгоритам"
)

plt.title(
    "Поређење тачности класификационих алгоритама"
)

plt.ylim(
    0,
    105
)

plt.grid(
    axis="y",
    alpha=0.3
)


for bar, value in zip(
    bars,
    values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2.0,
        bar.get_height()
        + 1.0,
        f"{value:.2f}%",
        ha="center",
        fontsize=9
    )


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_poredjenje_tacnosti_algoritama.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("=" * 100)
print("6.5.2 ANALIZA RANDOM FOREST KLASIFIKATORA")
print("=" * 100)


rf_model = trained_models[
    "Random Forest"
]

rf_pred = rf_model.predict(
    X_test
)

rf_train_pred = rf_model.predict(
    X_train
)


rf_train_accuracy = accuracy_score(
    y_train,
    rf_train_pred
)

rf_test_accuracy = accuracy_score(
    y_test,
    rf_pred
)


print(
    f"RF trening tacnost: "
    f"{rf_train_accuracy * 100:.2f}%"
)

print(
    f"RF test tacnost: "
    f"{rf_test_accuracy * 100:.2f}%"
)


report = classification_report(
    y_test,
    rf_pred,
    labels=CLASSES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "02_RF_classification_report.csv"
    )
)

cm = confusion_matrix(
    y_test,
    rf_pred,
    labels=CLASSES
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        CLASS_LABELS_CYR[
            c
        ]
        for c in CLASSES
    ]
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title(
    "Матрица конфузије Random Forest класификатора"
)

ax.set_xlabel(
    "Предвиђена класа"
)

ax.set_ylabel(
    "Стварна класа"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_RF_matrica_konfuzije.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


rf_summary = pd.DataFrame([{
    "train_accuracy": rf_train_accuracy,
    "test_accuracy": rf_test_accuracy,
    "difference_percentage_points":
        (
            rf_train_accuracy
            - rf_test_accuracy
        )
        * 100.0
}])

rf_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "02_RF_train_test_tacnost.csv"
    ),
    index=False
)


print()
print("=" * 100)
print("6.5.3 UTICAJ SNR-a NA TACNOST KLASIFIKACIJE")
print("=" * 100)


# Za SNR eksperiment koristi se zaseban, strogo balansiran test skup:
# 50 primera po klasi po SNR nivou.

SNR_TEST_PER_CLASS = 50

snr_test_X = []
snr_test_y = []
snr_test_values = []


for snr_db in SNR_VALUES:

    for modulation in CLASSES:

        for _ in range(
            SNR_TEST_PER_CLASS
        ):

            iq = generate_signal(
                modulation,
                snr_db
            )

            features = extract_final_features(
                iq
            )

            snr_test_X.append(
                features
            )

            snr_test_y.append(
                modulation
            )

            snr_test_values.append(
                snr_db
            )


snr_test_X = np.asarray(
    snr_test_X,
    dtype=float
)

snr_test_y = np.asarray(
    snr_test_y
)

snr_test_values = np.asarray(
    snr_test_values
)


rf_snr_predictions = rf_model.predict(
    snr_test_X
)

svm_model = trained_models[
    "SVM"
]

svm_snr_predictions = svm_model.predict(
    snr_test_X
)


snr_results = []


for snr_db in SNR_VALUES:

    mask = (
        snr_test_values
        == snr_db
    )


    rf_accuracy = accuracy_score(
        snr_test_y[
            mask
        ],
        rf_snr_predictions[
            mask
        ]
    )


    svm_accuracy = accuracy_score(
        snr_test_y[
            mask
        ],
        svm_snr_predictions[
            mask
        ]
    )


    snr_results.append({
        "snr_db": snr_db,
        "rf_accuracy": rf_accuracy,
        "svm_accuracy": svm_accuracy,
        "num_examples": int(
            np.sum(mask)
        )
    })


snr_df = pd.DataFrame(
    snr_results
)

snr_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "03_tacnost_po_SNR.csv"
    ),
    index=False
)


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    snr_df[
        "snr_db"
    ],
    snr_df[
        "rf_accuracy"
    ] * 100.0,
    marker="o",
    linewidth=2,
    label="Random Forest"
)

plt.plot(
    snr_df[
        "snr_db"
    ],
    snr_df[
        "svm_accuracy"
    ] * 100.0,
    marker="s",
    linewidth=2,
    label="SVM"
)

plt.xlabel(
    "Однос сигнал–шум SNR [dB]"
)

plt.ylabel(
    "Тачност [%]"
)

plt.title(
    "Утицај SNR-а на тачност класификације"
)

plt.xticks(
    SNR_VALUES
)

plt.ylim(
    0,
    105
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_tacnost_u_zavisnosti_od_SNR.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6.5.4 ROBUSNOST PO MODULACIONIM KLASAMA
# ============================================================

print()
print("=" * 100)
print("6.5.4 ROBUSNOST PO MODULACIONIM KLASAMA")
print("=" * 100)


def calculate_per_class_snr(
    true_labels,
    predicted_labels,
    snr_values
):

    rows = []

    for snr_db in SNR_VALUES:

        for modulation in CLASSES:

            mask = (
                (
                    snr_values
                    == snr_db
                )
                &
                (
                    true_labels
                    == modulation
                )
            )


            accuracy = accuracy_score(
                true_labels[
                    mask
                ],
                predicted_labels[
                    mask
                ]
            )


            rows.append({
                "snr_db": snr_db,
                "class": modulation,
                "accuracy": accuracy,
                "num_examples": int(
                    np.sum(mask)
                )
            })


    return pd.DataFrame(
        rows
    )


rf_class_snr_df = calculate_per_class_snr(
    snr_test_y,
    rf_snr_predictions,
    snr_test_values
)

svm_class_snr_df = calculate_per_class_snr(
    snr_test_y,
    svm_snr_predictions,
    snr_test_values
)


rf_class_snr_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "04_RF_tacnost_po_klasi_i_SNR.csv"
    ),
    index=False
)

svm_class_snr_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "04_SVM_tacnost_po_klasi_i_SNR.csv"
    ),
    index=False
)


plt.figure(
    figsize=(10, 6)
)

for modulation in CLASSES:

    temp = rf_class_snr_df[
        rf_class_snr_df[
            "class"
        ]
        == modulation
    ]

    plt.plot(
        temp[
            "snr_db"
        ],
        temp[
            "accuracy"
        ] * 100.0,
        marker="o",
        linewidth=2,
        label=CLASS_LABELS_CYR[
            modulation
        ]
    )


plt.xlabel(
    "Однос сигнал–шум SNR [dB]"
)

plt.ylabel(
    "Тачност [%]"
)

plt.title(
    "Тачност Random Forest класификатора по модулационим класама"
)

plt.xticks(
    SNR_VALUES
)

plt.ylim(
    0,
    105
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_RF_robusnost_po_klasama.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(
    figsize=(10, 6)
)

for modulation in CLASSES:

    temp = svm_class_snr_df[
        svm_class_snr_df[
            "class"
        ]
        == modulation
    ]

    plt.plot(
        temp[
            "snr_db"
        ],
        temp[
            "accuracy"
        ] * 100.0,
        marker="o",
        linewidth=2,
        label=CLASS_LABELS_CYR[
            modulation
        ]
    )


plt.xlabel(
    "Однос сигнал–шум SNR [dB]"
)

plt.ylabel(
    "Тачност [%]"
)

plt.title(
    "Тачност SVM класификатора по модулационим класама"
)

plt.xticks(
    SNR_VALUES
)

plt.ylim(
    0,
    105
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_SVM_robusnost_po_klasama.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("=" * 100)
print("6.5.5 UNAKRSNA VALIDACIJA I STABILNOST MODELA")
print("=" * 100)


cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)


scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro"
}


cv_models = {

    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
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
        ])
}


cv_rows = []
cv_fold_rows = []


for model_name, model in cv_models.items():

    print(
        "CV:",
        model_name
    )


    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )


    accuracy_values = scores[
        "test_accuracy"
    ]

    f1_values = scores[
        "test_f1_macro"
    ]


    cv_rows.append({
        "model": model_name,
        "mean_accuracy": np.mean(
            accuracy_values
        ),
        "std_accuracy": np.std(
            accuracy_values,
            ddof=1
        ),
        "min_accuracy": np.min(
            accuracy_values
        ),
        "max_accuracy": np.max(
            accuracy_values
        ),
        "mean_macro_f1": np.mean(
            f1_values
        )
    })


    for fold_number, (
        acc,
        f1
    ) in enumerate(
        zip(
            accuracy_values,
            f1_values
        ),
        start=1
    ):

        cv_fold_rows.append({
            "model": model_name,
            "fold": fold_number,
            "accuracy": acc,
            "macro_f1": f1
        })


cv_summary_df = pd.DataFrame(
    cv_rows
)

cv_fold_df = pd.DataFrame(
    cv_fold_rows
)


cv_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "05_CV_rezime.csv"
    ),
    index=False
)

cv_fold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "05_CV_rezultati_po_iteraciji.csv"
    ),
    index=False
)


plt.figure(
    figsize=(9, 6)
)

for model_name in [
    "Random Forest",
    "SVM"
]:

    temp = cv_fold_df[
        cv_fold_df[
            "model"
        ]
        == model_name
    ]

    plt.plot(
        temp[
            "fold"
        ],
        temp[
            "accuracy"
        ] * 100.0,
        marker="o",
        linewidth=2,
        label=model_name
    )


plt.xlabel(
    "Итерација унакрсне валидације"
)

plt.ylabel(
    "Тачност [%]"
)

plt.title(
    "Тачност модела по итерацијама 10-fold унакрсне валидације"
)

plt.xticks(
    range(
        1,
        11
    )
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_CV_tacnost_po_iteracijama.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


rf_cv_values = (
    cv_fold_df[
        cv_fold_df[
            "model"
        ]
        == "Random Forest"
    ][
        "accuracy"
    ].values
    * 100.0
)

svm_cv_values = (
    cv_fold_df[
        cv_fold_df[
            "model"
        ]
        == "SVM"
    ][
        "accuracy"
    ].values
    * 100.0
)


plt.figure(
    figsize=(7, 6)
)

plt.boxplot(
    [
        rf_cv_values,
        svm_cv_values
    ],
    labels=[
        "Random Forest",
        "SVM"
    ]
)

plt.ylabel(
    "Тачност [%]"
)

plt.title(
    "Расподела резултата 10-fold унакрсне валидације"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_CV_boxplot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("=" * 100)
print("6.5.6 ZNACAJ IZDVOJENIH OBELEZJA")
print("=" * 100)

builtin_importance = (
    rf_model.feature_importances_
)

importance_df = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "importance": builtin_importance
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(
    drop=True
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "06_RF_ugradjeni_znacaj_obelezja.csv"
    ),
    index=False
)


plt.figure(
    figsize=(10, 7)
)

plot_df = importance_df.iloc[::-1]

plt.barh(
    [
        FEATURE_LABELS_CYR[
            f
        ]
        for f in plot_df[
            "feature"
        ]
    ],
    plot_df[
        "importance"
    ] * 100.0
)

plt.xlabel(
    "Релативни значај [%]"
)

plt.ylabel(
    "Обележје"
)

plt.title(
    "Релативни значај обележја у Random Forest класификатору"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_RF_ugradjeni_znacaj_obelezja.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Racunanje permutation importance..."
)

permutation = permutation_importance(
    rf_model,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


permutation_df = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "mean_importance": permutation.importances_mean,
    "std_importance": permutation.importances_std
})

permutation_df = permutation_df.sort_values(
    "mean_importance",
    ascending=False
).reset_index(
    drop=True
)

permutation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "06_RF_permutacioni_znacaj_obelezja.csv"
    ),
    index=False
)


plot_df = permutation_df.iloc[::-1]


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    [
        FEATURE_LABELS_CYR[
            f
        ]
        for f in plot_df[
            "feature"
        ]
    ],
    plot_df[
        "mean_importance"
    ] * 100.0,
    xerr=(
        plot_df[
            "std_importance"
        ]
        * 100.0
    )
)

plt.xlabel(
    "Просечан пад тачности [процентни поени]"
)

plt.ylabel(
    "Обележје"
)

plt.title(
    "Пермутациони значај издвојених обележја"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_RF_permutacioni_znacaj_obelezja.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("=" * 100)
print("REZIME EKSPERIMENTA 6.5")
print("=" * 100)

print()
print("Ukupan broj primera:", len(y))

print(
    "Broj obelezja:",
    X.shape[1]
)

print(
    "Broj klasa:",
    len(CLASSES)
)

print(
    "SNR nivoi:",
    SNR_VALUES
)

print(
    "Primeraka po klasi i SNR nivou:",
    EXAMPLES_PER_CLASS_PER_SNR
)

print(
    "Glavni trening skup:",
    len(y_train)
)

print(
    "Glavni test skup:",
    len(y_test)
)


print()
print("-" * 100)
print("POREDJENJE MODELA")
print("-" * 100)

print(
    comparison_df[
        [
            "algorithm",
            "accuracy",
            "macro_f1",
            "balanced_accuracy"
        ]
    ].to_string(
        index=False
    )
)


print()
print("-" * 100)
print("RANDOM FOREST")
print("-" * 100)

print(
    f"Trening tacnost: "
    f"{rf_train_accuracy * 100:.2f}%"
)

print(
    f"Test tacnost: "
    f"{rf_test_accuracy * 100:.2f}%"
)


print()
print("-" * 100)
print("SNR ANALIZA")
print("-" * 100)

print(
    snr_df.to_string(
        index=False
    )
)


print()
print("-" * 100)
print("UNAKRSNA VALIDACIJA")
print("-" * 100)

print(
    cv_summary_df.to_string(
        index=False
    )
)


print()
print("-" * 100)
print("TOP 10 RF OBELEZJA")
print("-" * 100)

print(
    importance_df.head(
        10
    ).to_string(
        index=False
    )
)


print()
print("-" * 100)
print("TOP 10 PERMUTACIONIH OBELEZJA")
print("-" * 100)

print(
    permutation_df.head(
        10
    ).to_string(
        index=False
    )
)


print()
print("=" * 100)
print("SACUVANI REZULTATI")
print("=" * 100)

print(
    OUTPUT_DIR
)

print()
print("GOTOVO.")