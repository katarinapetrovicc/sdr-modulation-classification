import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

from features_final import extract_features


plt.rcParams["font.family"] = "DejaVu Sans"


BASE_DIR = Path("data") / "am" / "AM_125_925"

MODEL_FILE = Path("models") / "amc_model_final_16features_500ksps_10ms.pkl"

AM_FILES = [
    os.path.join(
        BASE_DIR,
        "AM_125925_event_20260917_195544.npz"
    ),
    os.path.join(
        BASE_DIR,
        "AM_125925_event_20260917_195720.npz"
    ),
    os.path.join(
        BASE_DIR,
        "AM_125925_event_20260917_200347.npz"
    )
]

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "AM_CLASSIFIER_VALIDATION"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


NUM_BLOCKS = 30
BLOCK_DURATION_S = 0.050

REAL_FIR_TAPS = 129
REAL_FIR_CUTOFF_HZ = 100_000

SIGNAL_HALF_WIDTH_HZ = 10_000
BACKGROUND_HALF_WIDTH_HZ = 100_000

DETECTION_THRESHOLD_DB = 8.0

NPERSEG = 4096
NOVERLAP = 2048


CLASS_ORDER = [
    "AM",
    "FM",
    "BPSK",
    "QPSK",
    "16-QAM"
]


TRANSMISSION_LABELS = {
    "195544": "Трансмисија 1",
    "195720": "Трансмисија 2",
    "200347": "Трансмисија 3"
}

print("=" * 100)
print("6.7.1 PROVERA KLASIFIKATORA NA REALNIM AM SIGNALIMA")
print("=" * 100)

with open(
    MODEL_FILE,
    "rb"
) as f:

    bundle = pickle.load(
        f
    )


model = bundle["model"]
feature_columns = bundle["feature_columns"]
fs_model = int(bundle["fs"])
final_samples = int(bundle["final_samples"])


print(
    "Model:",
    MODEL_FILE
)

print(
    "Broj obelezja:",
    len(feature_columns)
)

print(
    "Klase:",
    list(model.classes_)
)

print(
    "Fs modela:",
    fs_model
)

print(
    "Broj uzoraka modela:",
    final_samples
)


def get_transmission_id(
    file_path
):

    filename = os.path.basename(
        file_path
    )

    for transmission_id in [
        "195544",
        "195720",
        "200347"
    ]:

        if transmission_id in filename:

            return transmission_id

    return filename


def detection_score(
    block,
    fs_sdr,
    center_freq,
    target_freq
):

    block_dc = (
        block
        - np.mean(block)
    )

    freqs, psd = signal.welch(
        block_dc,
        fs=fs_sdr,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        return_onesided=False,
        scaling="density"
    )

    freqs = np.fft.fftshift(
        freqs
    )

    psd = np.fft.fftshift(
        psd
    )

    psd_db = (
        10.0
        * np.log10(
            psd
            + 1e-20
        )
    )

    rf_freqs = (
        center_freq
        + freqs
    )


    signal_mask = (
        np.abs(
            rf_freqs
            - target_freq
        )
        <= SIGNAL_HALF_WIDTH_HZ
    )

    background_mask = (
        (
            np.abs(
                rf_freqs
                - target_freq
            )
            <= BACKGROUND_HALF_WIDTH_HZ
        )
        &
        (
            np.abs(
                rf_freqs
                - target_freq
            )
            > SIGNAL_HALF_WIDTH_HZ
        )
    )


    signal_peak_db = float(
        np.max(
            psd_db[
                signal_mask
            ]
        )
    )

    background_median_db = float(
        np.median(
            psd_db[
                background_mask
            ]
        )
    )


    return (
        signal_peak_db
        - background_median_db
    )


def preprocess_real_block(
    block,
    fs_sdr,
    center_freq,
    target_freq
):

    # 1) uklanjanje DC komponente
    block = (
        block
        - np.mean(block)
    )


    # 2) frekvencijsko pomeranje ciljnog kanala u baseband
    freq_offset = (
        target_freq
        - center_freq
    )

    n = np.arange(
        len(block),
        dtype=np.float64
    )

    t = (
        n
        / fs_sdr
    )

    shifted = (
        block
        * np.exp(
            -1j
            * 2.0
            * np.pi
            * freq_offset
            * t
        )
    )


    # 3) FIR filtriranje
    real_fir = signal.firwin(
        REAL_FIR_TAPS,
        REAL_FIR_CUTOFF_HZ,
        fs=fs_sdr
    )

    filtered = signal.lfilter(
        real_fir,
        1.0,
        shifted
    )


    # 4) 1.024 MS/s -> 500 kS/s
    resampled = signal.resample_poly(
        filtered,
        125,
        256
    )


    # 5) odbacivanje tranzijenta
    resampled = resampled[
        200:
    ]


    # 6) centralnih 5000 uzoraka
    start_idx = (
        len(resampled)
        - final_samples
    ) // 2

    segment = resampled[
        start_idx:
        start_idx + final_samples
    ]


    if len(segment) != final_samples:

        raise ValueError(
            "Nije dobijeno tacno "
            f"{final_samples} uzoraka."
        )


    # 7) ista normalizacija snage kao u treningu
    power = np.mean(
        np.abs(segment) ** 2
    )

    if power > 1e-12:

        segment = (
            segment
            / np.sqrt(power)
        )


    return segment.astype(
        np.complex64
    )


# OBRADA SVA TRI REALNA AM SNIMKA

all_rows = []


for am_file in AM_FILES:

    print()
    print("=" * 100)
    print(
        "OBRADA:",
        am_file
    )
    print("=" * 100)


    transmission_id = get_transmission_id(
        am_file
    )

    transmission_label = (
        TRANSMISSION_LABELS[
            transmission_id
        ]
    )


    data = np.load(
        am_file,
        allow_pickle=True
    )


    samples_all = data[
        "samples"
    ].astype(
        np.complex128
    )

    fs_sdr = float(
        data["fs"]
    )

    center_freq = float(
        data["center_freq"]
    )

    target_freq = float(
        data["target_freq"]
    )

    pre_seconds = (
        float(
            data["pre_seconds"]
        )
        if "pre_seconds" in data.files
        else 2.0
    )


    print(
        f"Fs: "
        f"{fs_sdr / 1e6:.3f} MS/s"
    )

    print(
        f"Centar: "
        f"{center_freq / 1e6:.6f} MHz"
    )

    print(
        f"Cilj: "
        f"{target_freq / 1e6:.6f} MHz"
    )


    block_samples = int(
        round(
            fs_sdr
            * BLOCK_DURATION_S
        )
    )

    trigger_sample = int(
        round(
            pre_seconds
            * fs_sdr
        )
    )

    required_samples = (
        NUM_BLOCKS
        * block_samples
    )


    if (
        len(samples_all)
        - trigger_sample
        < required_samples
    ):

        raise ValueError(
            f"{transmission_label}: "
            "snimak nema dovoljno uzoraka "
            "za 30 blokova posle trigger-a."
        )


    analysis_samples = samples_all[
        trigger_sample:
        trigger_sample
        + required_samples
    ]


    for block_index in range(
        NUM_BLOCKS
    ):

        start = (
            block_index
            * block_samples
        )

        stop = (
            start
            + block_samples
        )

        block = analysis_samples[
            start:
            stop
        ]


        score_db = detection_score(
            block,
            fs_sdr,
            center_freq,
            target_freq
        )

        active = bool(
            score_db
            >= DETECTION_THRESHOLD_DB
        )


        segment = preprocess_real_block(
            block,
            fs_sdr,
            center_freq,
            target_freq
        )


        feature_dict = extract_features(
            segment,
            fs=fs_model
        )

        feature_vector = pd.DataFrame(
            [[
                feature_dict[
                    c
                ]
                for c
                in feature_columns
            ]],
            columns=feature_columns
        )


        prediction = model.predict(
            feature_vector
        )[0]


        probabilities = model.predict_proba(
            feature_vector
        )[0]


        probability_by_class = {
            cls: float(
                probabilities[i]
            )
            for i, cls
            in enumerate(
                model.classes_
            )
        }


        order = np.argsort(
            probabilities
        )[::-1]

        p_max = float(
            probabilities[
                order[0]
            ]
        )

        second_probability = float(
            probabilities[
                order[1]
            ]
        )

        margin = (
            p_max
            - second_probability
        )


        all_rows.append({
            "transmission_id":
                transmission_id,

            "transmission":
                transmission_label,

            "block":
                block_index + 1,

            "active":
                active,

            "detection_score_db":
                score_db,

            "prediction":
                prediction,

            "p_max":
                p_max,

            "margin":
                margin,

            "p_AM":
                probability_by_class.get(
                    "AM",
                    np.nan
                ),

            "p_FM":
                probability_by_class.get(
                    "FM",
                    np.nan
                ),

            "p_BPSK":
                probability_by_class.get(
                    "BPSK",
                    np.nan
                ),

            "p_QPSK":
                probability_by_class.get(
                    "QPSK",
                    np.nan
                ),

            "p_16QAM":
                probability_by_class.get(
                    "16-QAM",
                    np.nan
                )
        })


        status = (
            "AKTIVAN"
            if active
            else "neaktivan"
        )


        print(
            f"{transmission_id} | "
            f"blok {block_index + 1:02d} | "
            f"{status:9s} | "
            f"{prediction:7s} | "
            f"det={score_db:6.2f} dB | "
            f"p(AM)="
            f"{probability_by_class.get('AM', np.nan):.3f}"
        )


all_df = pd.DataFrame(
    all_rows
)

ALL_BLOCKS_CSV = os.path.join(
    OUTPUT_DIR,
    "01_AM_svi_blokovi.csv"
)

all_df.to_csv(
    ALL_BLOCKS_CSV,
    index=False
)


active_df = all_df[
    all_df["active"]
].copy()


ACTIVE_BLOCKS_CSV = os.path.join(
    OUTPUT_DIR,
    "02_AM_aktivni_blokovi.csv"
)

active_df.to_csv(
    ACTIVE_BLOCKS_CSV,
    index=False
)

summary_rows = []


for transmission_id in [
    "195544",
    "195720",
    "200347"
]:

    transmission_df = active_df[
        active_df[
            "transmission_id"
        ]
        == transmission_id
    ]


    total_active = len(
        transmission_df
    )

    correct_am = int(
        np.sum(
            transmission_df[
                "prediction"
            ]
            == "AM"
        )
    )

    accuracy_percent = (
        100.0
        * correct_am
        / total_active
        if total_active > 0
        else np.nan
    )


    counts = Counter(
        transmission_df[
            "prediction"
        ]
    )


    summary_rows.append({
        "transmission_id":
            transmission_id,

        "transmission":
            TRANSMISSION_LABELS[
                transmission_id
            ],

        "active_blocks":
            total_active,

        "AM":
            counts.get(
                "AM",
                0
            ),

        "FM":
            counts.get(
                "FM",
                0
            ),

        "BPSK":
            counts.get(
                "BPSK",
                0
            ),

        "QPSK":
            counts.get(
                "QPSK",
                0
            ),

        "16-QAM":
            counts.get(
                "16-QAM",
                0
            ),

        "accuracy_percent":
            accuracy_percent,

        "mean_p_AM":
            transmission_df[
                "p_AM"
            ].mean(),

        "mean_p_max":
            transmission_df[
                "p_max"
            ].mean(),

        "mean_margin":
            transmission_df[
                "margin"
            ].mean()
    })


summary_df = pd.DataFrame(
    summary_rows
)


total_active = len(
    active_df
)

total_correct_am = int(
    np.sum(
        active_df[
            "prediction"
        ]
        == "AM"
    )
)

overall_accuracy = (
    100.0
    * total_correct_am
    / total_active
)


summary_df.loc[
    len(summary_df)
] = {
    "transmission_id":
        "UKUPNO",

    "transmission":
        "Укупно",

    "active_blocks":
        total_active,

    "AM":
        int(
            np.sum(
                active_df[
                    "prediction"
                ]
                == "AM"
            )
        ),

    "FM":
        int(
            np.sum(
                active_df[
                    "prediction"
                ]
                == "FM"
            )
        ),

    "BPSK":
        int(
            np.sum(
                active_df[
                    "prediction"
                ]
                == "BPSK"
            )
        ),

    "QPSK":
        int(
            np.sum(
                active_df[
                    "prediction"
                ]
                == "QPSK"
            )
        ),

    "16-QAM":
        int(
            np.sum(
                active_df[
                    "prediction"
                ]
                == "16-QAM"
            )
        ),

    "accuracy_percent":
        overall_accuracy,

    "mean_p_AM":
        active_df[
            "p_AM"
        ].mean(),

    "mean_p_max":
        active_df[
            "p_max"
        ].mean(),

    "mean_margin":
        active_df[
            "margin"
        ].mean()
}


SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "03_AM_rezime_po_transmisijama.csv"
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)

print()
print("=" * 100)
print("REZIME REALNE AM VALIDACIJE")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print(
    f"UKUPNO AKTIVNIH BLOKOVA: "
    f"{total_active}"
)

print(
    f"ISPRAVNO AM: "
    f"{total_correct_am}"
)

print(
    f"UKUPNA USPESNOST: "
    f"{overall_accuracy:.2f}%"
)


# SLIKA 1
# RASPODELA PREDIKCIJA PO TRANSMISIJAMA

plot_summary = summary_df[
    summary_df[
        "transmission_id"
    ]
    != "UKUPNO"
].copy()


x = np.arange(
    len(
        plot_summary
    )
)

bottom = np.zeros(
    len(
        plot_summary
    )
)


plt.figure(
    figsize=(10, 6)
)


for cls in CLASS_ORDER:

    values = plot_summary[
        cls
    ].values

    plt.bar(
        x,
        values,
        bottom=bottom,
        label=cls
    )

    bottom = (
        bottom
        + values
    )


plt.xticks(
    x,
    plot_summary[
        "transmission"
    ]
)

plt.xlabel(
    "Независна AM трансмисија"
)

plt.ylabel(
    "Број активних блокова"
)

plt.title(
    "Расподела предвиђених класа по реалним AM трансмисијама"
)

plt.legend(
    title="Предвиђена класа"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

FIGURE_1 = os.path.join(
    OUTPUT_DIR,
    "01_AM_predikcije_po_transmisijama.png"
)

plt.savefig(
    FIGURE_1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# SLIKA 2
# P(AM) ZA SVE AKTIVNE BLOKOVE

active_plot = active_df.copy()

active_plot[
    "global_index"
] = np.arange(
    1,
    len(active_plot) + 1
)


plt.figure(
    figsize=(12, 6)
)


for transmission_id in [
    "195544",
    "195720",
    "200347"
]:

    temp = active_plot[
        active_plot[
            "transmission_id"
        ]
        == transmission_id
    ]

    plt.plot(
        temp[
            "global_index"
        ],
        temp[
            "p_AM"
        ],
        marker="o",
        linewidth=1.5,
        label=TRANSMISSION_LABELS[
            transmission_id
        ]
    )


# oznaka pogresno klasifikovanih aktivnih blokova
wrong_df = active_plot[
    active_plot[
        "prediction"
    ]
    != "AM"
]


if len(
    wrong_df
) > 0:

    plt.scatter(
        wrong_df[
            "global_index"
        ],
        wrong_df[
            "p_AM"
        ],
        s=120,
        marker="x",
        linewidths=2.5,
        label="Погрешна одлука"
    )


plt.xlabel(
    "Редни број активног блока"
)

plt.ylabel(
    "Излазна вероватноћа класе AM"
)

plt.title(
    "Излазна вероватноћа AM класе за активне блокове реалних трансмисија"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

FIGURE_2 = os.path.join(
    OUTPUT_DIR,
    "02_AM_verovatnoca_po_aktivnim_blokovima.png"
)

plt.savefig(
    FIGURE_2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# SLIKA 3
# DETEKCIONI SCORE I PRAG AKTIVNOSTI
plt.figure(
    figsize=(12, 6)
)


for transmission_id in [
    "195544",
    "195720",
    "200347"
]:

    temp = all_df[
        all_df[
            "transmission_id"
        ]
        == transmission_id
    ].copy()

    plt.plot(
        temp[
            "block"
        ],
        temp[
            "detection_score_db"
        ],
        marker="o",
        linewidth=1.5,
        label=TRANSMISSION_LABELS[
            transmission_id
        ]
    )


plt.axhline(
    DETECTION_THRESHOLD_DB,
    linestyle="--",
    linewidth=1.5,
    label="Праг активности = 8 dB"
)

plt.xlabel(
    "Редни број блока"
)

plt.ylabel(
    "Разлика сигнал–локална позадина [dB]"
)

plt.title(
    "Детекција активности у реалним AM трансмисијама"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

FIGURE_3 = os.path.join(
    OUTPUT_DIR,
    "03_AM_detekcija_aktivnih_blokova.png"
)

plt.savefig(
    FIGURE_3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("=" * 100)
print("SACUVANI FAJLOVI")
print("=" * 100)

for file_path in [
    ALL_BLOCKS_CSV,
    ACTIVE_BLOCKS_CSV,
    SUMMARY_CSV,
    FIGURE_1,
    FIGURE_2,
    FIGURE_3
]:

    print(
        file_path
    )



print()
print("GOTOVO.")