import os
import glob
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

from features_final import extract_features

plt.rcParams["font.family"] = "DejaVu Sans"


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "amc_model_final_16features_500ksps_10ms.pkl"
)

FM_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "fm"
    / "FM_98_7047"
)

OUTPUT_DIR = os.path.join(
    FM_FOLDER,
    "FM_CLASSIFIER_VALIDATION"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# PARAMETRI FINALNE FM VALIDACIJE

NUM_BLOCKS = 30

REAL_FIR_TAPS = 129
REAL_FIR_CUTOFF_HZ = 100_000

CLASS_ORDER = [
    "FM",
    "AM",
    "BPSK",
    "QPSK",
    "16-QAM"
]

print("=" * 100)
print("6.7.2 PROVERA KLASIFIKATORA NA REALNIM FM SIGNALIMA")
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


if len(feature_columns) != 16:

    raise ValueError(
        "Finalni model ne koristi 16 obelezja."
    )


# PRONALAZENJE SVIH 9 REALNIH FM SNIMAKA

fm_files = sorted(
    glob.glob(
        os.path.join(
            FM_FOLDER,
            "FM_*_event_*.npz"
        )
    )
)


if len(fm_files) == 0:

    raise FileNotFoundError(
        "Nijedan FM .npz fajl nije pronadjen."
    )


print()
print(
    "Pronadjeno FM snimaka:",
    len(fm_files)
)


if len(fm_files) != 9:

    print(
        "UPOZORENJE: Ocekivano je 9 finalnih FM snimaka, "
        f"a pronadjeno je {len(fm_files)}."
    )


def preprocess_full_fm_recording(
    samples,
    fs_sdr,
    center_freq,
    station_freq
):

    """
    Obrada jednog realnog FM snimka:
    - uklanjanje DC komponente
    - pomeranje ciljne stanice u baseband
    - FIR LPF 100 kHz
    - promena Fs sa 2.4 MS/s na 500 kS/s
    - odbacivanje tranzijenta
    """

    # 1) DC uklanjanje
    x = (
        samples
        - np.mean(samples)
    )


    # 2) frekvencijsko pomeranje
    freq_offset = (
        station_freq
        - center_freq
    )

    n = np.arange(
        len(x),
        dtype=np.float64
    )

    mixer = np.exp(
        -1j
        * 2.0
        * np.pi
        * freq_offset
        * n
        / fs_sdr
    )

    shifted = (
        x
        * mixer
    )


    # 3) FIR filtriranje
    fir_coeff = signal.firwin(
        REAL_FIR_TAPS,
        REAL_FIR_CUTOFF_HZ,
        fs=fs_sdr
    )

    filtered = signal.lfilter(
        fir_coeff,
        1.0,
        shifted
    )


    # 4) 2.4 MS/s -> 500 kS/s
    #
    # 2.4 MHz * 5 / 24 = 500 kHz
    resampled = signal.resample_poly(
        filtered,
        5,
        24
    )


    # 5) odbacivanje pocetnog tranzijenta
    transient = 200

    if len(resampled) <= transient:

        raise ValueError(
            "Snimak je prekratak nakon resamplovanja."
        )

    resampled = resampled[
        transient:
    ]


    return resampled.astype(
        np.complex64
    )


def normalize_segment(
    segment
):

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


def test_one_file(
    file_path
):

    data = np.load(
        file_path,
        allow_pickle=True
    )


    samples = data[
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

    station_freq = float(
        data["station_freq"]
    )


    print()
    print("-" * 100)

    print(
        "Fajl:",
        os.path.basename(
            file_path
        )
    )

    print(
        f"Fs SDR-a: "
        f"{fs_sdr / 1e6:.3f} MS/s"
    )

    print(
        f"Centar: "
        f"{center_freq / 1e6:.6f} MHz"
    )

    print(
        f"FM signal: "
        f"{station_freq / 1e6:.6f} MHz"
    )

    resampled = preprocess_full_fm_recording(
        samples,
        fs_sdr,
        center_freq,
        station_freq
    )


    # CENTRALNIH 30 x 5000 UZORAKA
    # 30 blokova x 10 ms = 300 ms

    needed_samples = (
        NUM_BLOCKS
        * final_samples
    )


    if len(resampled) < needed_samples:

        raise ValueError(
            "Nema dovoljno resamplovanih uzoraka "
            "za 30 blokova."
        )


    start_idx = (
        len(resampled)
        - needed_samples
    ) // 2


    test_signal = resampled[
        start_idx:
        start_idx + needed_samples
    ]


    rows = []

    for block_idx in range(
        NUM_BLOCKS
    ):

        start = (
            block_idx
            * final_samples
        )

        stop = (
            start
            + final_samples
        )


        iq_segment = test_signal[
            start:
            stop
        ]


        if len(iq_segment) != final_samples:

            raise ValueError(
                "Blok nema tacno "
                f"{final_samples} uzoraka."
            )


        # normalizacija snage
        iq_segment = normalize_segment(
            iq_segment
        )


        # 16 finalnih obelezja
        feature_dict = extract_features(
            iq_segment,
            fs=fs_model
        )


        feature_vector = pd.DataFrame(
            [[
                feature_dict[
                    feature_name
                ]
                for feature_name
                in feature_columns
            ]],
            columns=feature_columns
        )


        # predikcija
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


        rows.append({
            "file":
                os.path.basename(
                    file_path
                ),

            "station_mhz":
                station_freq / 1e6,

            "block":
                block_idx + 1,

            "prediction":
                prediction,

            "p_max":
                p_max,

            "margin":
                margin,

            "p_FM":
                probability_by_class.get(
                    "FM",
                    np.nan
                ),

            "p_AM":
                probability_by_class.get(
                    "AM",
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


        print(
            f"blok {block_idx + 1:02d} | "
            f"{prediction:7s} | "
            f"p(FM)="
            f"{probability_by_class.get('FM', np.nan):.3f} | "
            f"pmax={p_max:.3f} | "
            f"margin={margin:.3f}"
        )


    detail_df = pd.DataFrame(
        rows
    )


    counts = Counter(
        detail_df[
            "prediction"
        ]
    )


    fm_correct = counts.get(
        "FM",
        0
    )


    accuracy_percent = (
        100.0
        * fm_correct
        / NUM_BLOCKS
    )


    return {
        "file":
            os.path.basename(
                file_path
            ),

        "station_mhz":
            station_freq / 1e6,

        "total_blocks":
            NUM_BLOCKS,

        "FM":
            counts.get(
                "FM",
                0
            ),

        "AM":
            counts.get(
                "AM",
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

        "mean_pFM":
            float(
                detail_df[
                    "p_FM"
                ].mean()
            ),

        "mean_pmax":
            float(
                detail_df[
                    "p_max"
                ].mean()
            ),

        "mean_margin":
            float(
                detail_df[
                    "margin"
                ].mean()
            ),

        "details":
            detail_df
    }


# OBRADA SVIH FM SNIMAKA
all_detail_dfs = []
summary_rows = []


for i, file_path in enumerate(
    fm_files,
    start=1
):

    print()
    print("=" * 100)

    print(
        f"SNIMAK {i}/{len(fm_files)}"
    )

    print("=" * 100)


    result = test_one_file(
        file_path
    )


    summary_rows.append({
        "file":
            result[
                "file"
            ],

        "station_mhz":
            result[
                "station_mhz"
            ],

        "total_blocks":
            result[
                "total_blocks"
            ],

        "FM":
            result[
                "FM"
            ],

        "AM":
            result[
                "AM"
            ],

        "BPSK":
            result[
                "BPSK"
            ],

        "QPSK":
            result[
                "QPSK"
            ],

        "16-QAM":
            result[
                "16-QAM"
            ],

        "accuracy_percent":
            result[
                "accuracy_percent"
            ],

        "mean_pFM":
            result[
                "mean_pFM"
            ],

        "mean_pmax":
            result[
                "mean_pmax"
            ],

        "mean_margin":
            result[
                "mean_margin"
            ]
    })


    all_detail_dfs.append(
        result[
            "details"
        ]
    )


    print()
    print(
        f"FM: "
        f"{result['FM']}/{NUM_BLOCKS}"
    )

    print(
        f"Tacnost: "
        f"{result['accuracy_percent']:.2f}%"
    )

    print(
        f"Prosecan p(FM): "
        f"{result['mean_pFM']:.4f}"
    )


# SVIH 270 BLOKOVA
all_blocks_df = pd.concat(
    all_detail_dfs,
    ignore_index=True
)


ALL_BLOCKS_CSV = os.path.join(
    OUTPUT_DIR,
    "01_FM_svi_blokovi.csv"
)

all_blocks_df.to_csv(
    ALL_BLOCKS_CSV,
    index=False
)


recording_summary_df = pd.DataFrame(
    summary_rows
)


RECORDING_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "02_FM_rezime_po_snimcima.csv"
)

recording_summary_df.to_csv(
    RECORDING_SUMMARY_CSV,
    index=False
)


station_rows = []


for station_mhz, group in all_blocks_df.groupby(
    "station_mhz"
):

    counts = Counter(
        group[
            "prediction"
        ]
    )

    total_blocks_station = len(
        group
    )

    fm_correct = counts.get(
        "FM",
        0
    )


    station_rows.append({
        "station_mhz":
            station_mhz,

        "total_blocks":
            total_blocks_station,

        "FM":
            counts.get(
                "FM",
                0
            ),

        "AM":
            counts.get(
                "AM",
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
            (
                100.0
                * fm_correct
                / total_blocks_station
            ),

        "mean_pFM":
            float(
                group[
                    "p_FM"
                ].mean()
            ),

        "mean_pmax":
            float(
                group[
                    "p_max"
                ].mean()
            ),

        "mean_margin":
            float(
                group[
                    "margin"
                ].mean()
            )
    })


station_summary_df = pd.DataFrame(
    station_rows
).sort_values(
    "station_mhz"
).reset_index(
    drop=True
)

overall_counts = Counter(
    all_blocks_df[
        "prediction"
    ]
)

total_blocks = len(
    all_blocks_df
)

total_correct = overall_counts.get(
    "FM",
    0
)

overall_accuracy = (
    100.0
    * total_correct
    / total_blocks
)


overall_row = {
    "station_mhz":
        np.nan,

    "total_blocks":
        total_blocks,

    "FM":
        overall_counts.get(
            "FM",
            0
        ),

    "AM":
        overall_counts.get(
            "AM",
            0
        ),

    "BPSK":
        overall_counts.get(
            "BPSK",
            0
        ),

    "QPSK":
        overall_counts.get(
            "QPSK",
            0
        ),

    "16-QAM":
        overall_counts.get(
            "16-QAM",
            0
        ),

    "accuracy_percent":
        overall_accuracy,

    "mean_pFM":
        float(
            all_blocks_df[
                "p_FM"
            ].mean()
        ),

    "mean_pmax":
        float(
            all_blocks_df[
                "p_max"
            ].mean()
        ),

    "mean_margin":
        float(
            all_blocks_df[
                "margin"
            ].mean()
        )
}


station_table_df = pd.concat(
    [
        station_summary_df,
        pd.DataFrame(
            [overall_row]
        )
    ],
    ignore_index=True
)


STATION_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "03_FM_rezime_po_frekvencijama.csv"
)

station_table_df.to_csv(
    STATION_SUMMARY_CSV,
    index=False
)

print()
print("=" * 100)
print("REZIME PO 9 NEZAVISNIH FM SNIMAKA")
print("=" * 100)

print(
    recording_summary_df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("REZIME PO FREKVENCIJAMA")
print("=" * 100)

print(
    station_table_df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("UKUPNA REALNA FM VALIDACIJA")
print("=" * 100)

print(
    f"Ukupno blokova: "
    f"{total_blocks}"
)

print(
    f"Ispravno FM: "
    f"{total_correct}"
)

print(
    f"Ukupna uspesnost: "
    f"{overall_accuracy:.2f}%"
)

print(
    f"AM greske: "
    f"{overall_counts.get('AM', 0)}"
)

print(
    f"BPSK greske: "
    f"{overall_counts.get('BPSK', 0)}"
)

print(
    f"QPSK greske: "
    f"{overall_counts.get('QPSK', 0)}"
)

print(
    f"16-QAM greske: "
    f"{overall_counts.get('16-QAM', 0)}"
)


# SLIKA 1
# RASPODELA PREDIKCIJA PO FREKVENCIJAMA

plot_df = station_summary_df.copy()

x = np.arange(
    len(plot_df)
)

labels = [
    f"{freq:.4f} MHz"
    for freq
    in plot_df[
        "station_mhz"
    ]
]

bottom = np.zeros(
    len(plot_df)
)


plt.figure(
    figsize=(10, 6)
)


for cls in CLASS_ORDER:

    values = plot_df[
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
    labels
)

plt.xlabel(
    "Фреквенција реалног FM сигнала"
)

plt.ylabel(
    "Број анализираних блокова"
)

plt.title(
    "Расподела предвиђених класа по реалним FM сигналима"
)

plt.ylim(
    0,
    95
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.legend(
    title="Предвиђена класа"
)

plt.tight_layout()


FIGURE_1 = os.path.join(
    OUTPUT_DIR,
    "01_FM_predikcije_po_frekvencijama.png"
)

plt.savefig(
    FIGURE_1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# SLIKA 2
# PROSECNA P(FM) PO FREKVENCIJAMA
plt.figure(
    figsize=(10, 6)
)


bars = plt.bar(
    x,
    plot_df[
        "mean_pFM"
    ]
)


plt.xticks(
    x,
    labels
)

plt.xlabel(
    "Фреквенција реалног FM сигнала"
)

plt.ylabel(
    "Просечна излазна вероватноћа класе FM"
)

plt.title(
    "Просечна вероватноћа FM класе по посматраним фреквенцијама"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    axis="y",
    alpha=0.3
)


for bar, value in zip(
    bars,
    plot_df[
        "mean_pFM"
    ]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2.0,
        bar.get_height()
        + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=10
    )


plt.tight_layout()


FIGURE_2 = os.path.join(
    OUTPUT_DIR,
    "02_FM_verovatnoca_po_frekvencijama.png"
)

plt.savefig(
    FIGURE_2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# SLIKA 3 
# P(FM) ZA SVIH 270 BLOKOVA

plot_blocks = all_blocks_df.copy()

plot_blocks[
    "global_index"
] = np.arange(
    1,
    len(plot_blocks) + 1
)


plt.figure(
    figsize=(13, 6)
)


for station_mhz in sorted(
    plot_blocks[
        "station_mhz"
    ].unique()
):

    temp = plot_blocks[
        plot_blocks[
            "station_mhz"
        ]
        == station_mhz
    ]

    plt.plot(
        temp[
            "global_index"
        ],
        temp[
            "p_FM"
        ],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label=(
            f"{station_mhz:.4f} MHz"
        )
    )


wrong_df = plot_blocks[
    plot_blocks[
        "prediction"
    ]
    != "FM"
]


if len(wrong_df) > 0:

    plt.scatter(
        wrong_df[
            "global_index"
        ],
        wrong_df[
            "p_FM"
        ],
        s=80,
        marker="x",
        linewidths=2.0,
        label="Погрешна одлука"
    )


plt.xlabel(
    "Редни број анализираног блока"
)

plt.ylabel(
    "Излазна вероватноћа класе FM"
)

plt.title(
    "Излазна вероватноћа FM класе за блокове реалних FM сигнала"
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


FIGURE_3 = os.path.join(
    OUTPUT_DIR,
    "03_FM_verovatnoca_po_blokovima.png"
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
    RECORDING_SUMMARY_CSV,
    STATION_SUMMARY_CSV,
    FIGURE_1,
    FIGURE_2,
    FIGURE_3
]:

    print(
        file_path
    )


print()
print("GOTOVO.")