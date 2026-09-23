# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FM_FILE = (
    PROJECT_ROOT
    / "data"
    / "fm"
    / "FM_98_7047"
    / "FM_98_7047_event_20260917_204145.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "spectral_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

plt.rcParams["font.family"] = "DejaVu Sans"


SEGMENT_START_SECONDS = 2.0
SEGMENT_DURATION_SECONDS = 2.0

NUM_REPEATS = 20

BASE_NPERSEG = 4096
BASE_NOVERLAP = BASE_NPERSEG // 2

NPERSEG_VALUES = [
    512,
    1024,
    2048,
    4096,
    8192
]

WINDOWS = [
    "hann",
    "hamming",
    "blackman",
    "boxcar"
]


def db10(x):

    return (
        10.0
        * np.log10(
            np.maximum(
                x,
                1e-20
            )
        )
    )


def benchmark_function(
    function,
    repeats=20
):

    # warm-up
    function()

    times_ms = []

    for _ in range(
        repeats
    ):

        start_time = time.perf_counter()

        function()

        stop_time = time.perf_counter()

        times_ms.append(
            (
                stop_time
                - start_time
            )
            * 1000.0
        )

    times_ms = np.array(
        times_ms
    )

    return (
        float(
            np.mean(
                times_ms
            )
        ),
        float(
            np.std(
                times_ms,
                ddof=1
            )
        )
    )


def create_zoom_axis(
    main_axis,
    xlim_zoom,
    ylim_zoom,
    bbox_to_anchor=(0.07, 0.07, 1, 1),
    width="33%",
    height="33%"
):

    zoom_axis = inset_axes(
        main_axis,
        width=width,
        height=height,
        loc="lower left",
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=main_axis.transAxes,
        borderpad=1.0
    )

    zoom_axis.set_xlim(
        xlim_zoom
    )

    zoom_axis.set_ylim(
        ylim_zoom
    )

    zoom_axis.grid(
        True,
        alpha=0.3
    )

    zoom_axis.tick_params(
        axis="both",
        labelsize=8
    )

    return zoom_axis


data = np.load(
    FM_FILE,
    allow_pickle=True
)

x_all = data[
    "samples"
].astype(
    np.complex64
)

fs = float(
    data["fs"]
)

center_freq = float(
    data["center_freq"]
)

station_freq = float(
    data["station_freq"]
)

station_freq_mhz = (
    station_freq
    / 1e6
)

duration_total = (
    len(x_all)
    / fs
)


print("=" * 90)
print("POREDJENJE POSTUPAKA SPEKTRALNE ANALIZE")
print("=" * 90)

print(
    f"Ukupno trajanje snimka: "
    f"{duration_total:.2f} s"
)

print(
    f"Fs: "
    f"{fs / 1e6:.3f} MS/s"
)

print(
    f"Centar SDR-a: "
    f"{center_freq / 1e6:.6f} MHz"
)

print(
    f"FM signal: "
    f"{station_freq_mhz:.6f} MHz"
)

start_sample = int(
    SEGMENT_START_SECONDS
    * fs
)

num_samples = int(
    SEGMENT_DURATION_SECONDS
    * fs
)

stop_sample = (
    start_sample
    + num_samples
)

if stop_sample > len(x_all):

    raise ValueError(
        "Izabrani segment izlazi van granica snimka."
    )


x = x_all[
    start_sample:
    stop_sample
].copy()

# uklanjanje DC komponente
x = (
    x
    - np.mean(x)
)


print(
    f"Korisceni interval: "
    f"{SEGMENT_START_SECONDS:.2f} - "
    f"{SEGMENT_START_SECONDS + SEGMENT_DURATION_SECONDS:.2f} s"
)

print(
    f"Broj kompleksnih I/Q uzoraka: "
    f"{len(x):,}"
)


print()
print("=" * 90)
print("6.4.1 FFT VS WELCH")
print("=" * 90)


def calculate_fft():

    return np.fft.fft(
        x
    )


fft_mean_ms, fft_std_ms = benchmark_function(
    calculate_fft,
    NUM_REPEATS
)


# stvarni FFT
fft_values = np.fft.fft(
    x
)

fft_values = np.fft.fftshift(
    fft_values
)

fft_freq = np.fft.fftfreq(
    len(x),
    d=1.0 / fs
)

fft_freq = np.fft.fftshift(
    fft_freq
)

fft_psd = (
    np.abs(
        fft_values
    ) ** 2
    / (
        fs
        * len(x)
    )
)

fft_psd_db = db10(
    fft_psd
)

fft_rf_freq = (
    center_freq
    + fft_freq
)


# WELCH

def calculate_welch():

    return signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=BASE_NPERSEG,
        noverlap=BASE_NOVERLAP,
        nfft=BASE_NPERSEG,
        return_onesided=False,
        scaling="density"
    )


welch_mean_ms, welch_std_ms = benchmark_function(
    calculate_welch,
    NUM_REPEATS
)


welch_freq, welch_psd = calculate_welch()

welch_freq = np.fft.fftshift(
    welch_freq
)

welch_psd = np.fft.fftshift(
    welch_psd
)

welch_psd_db = db10(
    welch_psd
)

welch_rf_freq = (
    center_freq
    + welch_freq
)


plot_width = 300e3

fft_mask = (
    np.abs(
        fft_rf_freq
        - station_freq
    )
    <= plot_width
)

welch_mask = (
    np.abs(
        welch_rf_freq
        - station_freq
    )
    <= plot_width
)


plt.figure(
    figsize=(12, 6)
)

ax = plt.gca()

ax.plot(
    fft_rf_freq[
        fft_mask
    ] / 1e6,
    fft_psd_db[
        fft_mask
    ],
    linewidth=0.6,
    alpha=0.55,
    label="FFT"
)

ax.plot(
    welch_rf_freq[
        welch_mask
    ] / 1e6,
    welch_psd_db[
        welch_mask
    ],
    linewidth=2.2,
    label="Welch"
)

ax.axvline(
    station_freq_mhz,
    linestyle="--",
    linewidth=1.5,
    label=f"{station_freq_mhz:.4f} MHz"
)

ax.set_xlabel(
    "Фреквенција [MHz]",
    fontsize=14
)

ax.set_ylabel(
    "PSD [dB/Hz]",
    fontsize=14
)

ax.set_title(
    "Поређење FFT и Welch спектралне анализе реалног FM сигнала",
    fontsize=16
)

ax.grid(
    alpha=0.3
)

ax.legend(
    fontsize=11
)

plt.tight_layout()

FFT_WELCH_FIGURE = os.path.join(
    OUTPUT_DIR,
    "01_FFT_vs_Welch.png"
)

plt.savefig(
    FFT_WELCH_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


fft_welch_results = pd.DataFrame({
    "postupak": [
        "FFT",
        "Welch"
    ],
    "mean_time_ms": [
        fft_mean_ms,
        welch_mean_ms
    ],
    "std_time_ms": [
        fft_std_ms,
        welch_std_ms
    ]
})


FFT_WELCH_CSV = os.path.join(
    OUTPUT_DIR,
    "01_FFT_vs_Welch_vreme.csv"
)

fft_welch_results.to_csv(
    FFT_WELCH_CSV,
    index=False
)

print(
    fft_welch_results.to_string(
        index=False
    )
)

print()
print("=" * 90)
print("6.4.2 UTICAJ DUZINE WELCH SEGMENTA")
print("=" * 90)


segment_results = []
segment_curves = []


for nperseg in NPERSEG_VALUES:

    noverlap = (
        nperseg
        // 2
    )


    def calculate_this_welch():

        return signal.welch(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg,
            return_onesided=False,
            scaling="density"
        )


    mean_ms, std_ms = benchmark_function(
        calculate_this_welch,
        NUM_REPEATS
    )


    freq, pxx = calculate_this_welch()

    freq = np.fft.fftshift(
        freq
    )

    pxx = np.fft.fftshift(
        pxx
    )

    pxx_db = db10(
        pxx
    )

    rf_freq = (
        center_freq
        + freq
    )

    rf_freq_mhz = (
        rf_freq
        / 1e6
    )


    search_mask = (
        np.abs(
            rf_freq
            - station_freq
        )
        <= 150e3
    )


    local_freq = rf_freq[
        search_mask
    ]

    local_psd = pxx_db[
        search_mask
    ]


    max_idx = np.argmax(
        local_psd
    )

    peak_freq = float(
        local_freq[
            max_idx
        ]
    )

    peak_psd = float(
        local_psd[
            max_idx
        ]
    )


    resolution_hz = (
        fs
        / nperseg
    )


    segment_results.append({
        "nperseg": nperseg,
        "noverlap": noverlap,
        "resolution_hz": resolution_hz,
        "peak_freq_mhz": peak_freq / 1e6,
        "peak_psd_db_hz": peak_psd,
        "mean_time_ms": mean_ms,
        "std_time_ms": std_ms
    })


    graph_mask = (
        np.abs(
            rf_freq
            - station_freq
        )
        <= 200e3
    )


    segment_curves.append({
        "nperseg": nperseg,
        "resolution_hz": resolution_hz,
        "freqs_mhz": rf_freq_mhz[
            graph_mask
        ],
        "psd_db": pxx_db[
            graph_mask
        ]
    })


plt.figure(
    figsize=(12, 7)
)

ax = plt.gca()


for item in segment_curves:

    ax.plot(
        item[
            "freqs_mhz"
        ],
        item[
            "psd_db"
        ],
        linewidth=1.6,
        label=(
            f"N = {item['nperseg']}, "
            f"Δf = {item['resolution_hz']:.1f} Hz"
        )
    )


ax.axvline(
    station_freq_mhz,
    linestyle="--",
    linewidth=1.5,
    label=f"{station_freq_mhz:.4f} MHz"
)

ax.set_xlabel(
    "Фреквенција [MHz]",
    fontsize=14
)

ax.set_ylabel(
    "PSD [dB/Hz]",
    fontsize=14
)

ax.set_title(
    "Утицај дужине сегмента на Welch-ову спектралну процену",
    fontsize=16
)

ax.grid(
    alpha=0.3
)

ax.legend(
    fontsize=9,
    loc="upper right"
)


zoom_x_min = (
    station_freq_mhz
    - 0.012
)

zoom_x_max = (
    station_freq_mhz
    + 0.012
)


zoom_y_values = []


for item in segment_curves:

    zoom_mask = (
        (
            item[
                "freqs_mhz"
            ]
            >= zoom_x_min
        )
        &
        (
            item[
                "freqs_mhz"
            ]
            <= zoom_x_max
        )
    )

    zoom_y_values.extend(
        item[
            "psd_db"
        ][
            zoom_mask
        ]
    )


zoom_y_values = np.array(
    zoom_y_values
)


zoom_y_min = float(
    np.min(
        zoom_y_values
    )
    - 0.4
)

zoom_y_max = float(
    np.max(
        zoom_y_values
    )
    + 0.4
)


axins = create_zoom_axis(
    ax,
    xlim_zoom=(
        zoom_x_min,
        zoom_x_max
    ),
    ylim_zoom=(
        zoom_y_min,
        zoom_y_max
    ),
    bbox_to_anchor=(
        0.06,
        0.06,
        1,
        1
    ),
    width="34%",
    height="34%"
)


for item in segment_curves:

    axins.plot(
        item[
            "freqs_mhz"
        ],
        item[
            "psd_db"
        ],
        linewidth=1.3
    )


axins.axvline(
    station_freq_mhz,
    linestyle="--",
    linewidth=1.2
)


mark_inset(
    ax,
    axins,
    loc1=2,
    loc2=4,
    fc="none",
    ec="gray",
    lw=1.0
)


plt.tight_layout()


SEGMENT_FIGURE = os.path.join(
    OUTPUT_DIR,
    "02_Welch_duzina_segmenta.png"
)

plt.savefig(
    SEGMENT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


segment_df = pd.DataFrame(
    segment_results
)


SEGMENT_CSV = os.path.join(
    OUTPUT_DIR,
    "02_Welch_duzina_segmenta.csv"
)

segment_df.to_csv(
    SEGMENT_CSV,
    index=False
)


print(
    segment_df.to_string(
        index=False
    )
)

print()
print("=" * 90)
print("6.4.3 UTICAJ PROZORSKE FUNKCIJE")
print("=" * 90)

window_results = []
window_curves = []


for window_name in WINDOWS:

    def calculate_window_welch():

        f_temp, pxx_temp = signal.welch(
            x,
            fs=fs,
            window=window_name,
            nperseg=BASE_NPERSEG,
            noverlap=BASE_NOVERLAP,
            nfft=BASE_NPERSEG,
            return_onesided=False,
            scaling="density"
        )

        return f_temp, pxx_temp

    mean_ms, std_ms = benchmark_function(
        calculate_window_welch,
        NUM_REPEATS
    )


    freq, pxx = calculate_window_welch()

    freq = np.asarray(
        freq
    ).ravel()

    pxx = np.asarray(
        pxx
    ).ravel()


    # FFT shift
    freq = np.fft.fftshift(
        freq
    )

    pxx = np.fft.fftshift(
        pxx
    )


    # PSD u dB/Hz
    pxx_db = db10(
        pxx
    )

    pxx_db = np.asarray(
        pxx_db
    ).ravel()


    # apsolutna RF frekvencija
    rf_freq = (
        center_freq
        + freq
    )

    rf_freq = np.asarray(
        rf_freq
    ).ravel()

    rf_freq_mhz = (
        rf_freq
        / 1e6
    )


    if len(rf_freq) != len(pxx_db):

        raise ValueError(
            f"Dimenzije se ne poklapaju za prozor {window_name}: "
            f"rf_freq={len(rf_freq)}, pxx_db={len(pxx_db)}"
        )


    search_mask = (
        np.abs(
            rf_freq
            - station_freq
        )
        <= 150e3
    )

    local_freq = rf_freq[
        search_mask
    ]

    local_psd = pxx_db[
        search_mask
    ]


    max_idx = np.argmax(
        local_psd
    )

    peak_freq = float(
        local_freq[
            max_idx
        ]
    )

    peak_psd = float(
        local_psd[
            max_idx
        ]
    )


    window_results.append({
        "window": window_name,
        "peak_freq_mhz": peak_freq / 1e6,
        "peak_psd_db_hz": peak_psd,
        "mean_time_ms": mean_ms,
        "std_time_ms": std_ms
    })


    graph_mask = (
        np.abs(
            rf_freq
            - station_freq
        )
        <= 200e3
    )

    window_curves.append({
        "window": window_name,
        "freqs_mhz": rf_freq_mhz[
            graph_mask
        ].copy(),
        "psd_db": pxx_db[
            graph_mask
        ].copy()
    })


plt.figure(
    figsize=(12, 7)
)

ax = plt.gca()


for item in window_curves:

    ax.plot(
        item["freqs_mhz"],
        item["psd_db"],
        linewidth=1.7,
        label=item["window"].capitalize()
    )


ax.axvline(
    station_freq_mhz,
    linestyle="--",
    linewidth=1.5,
    label=f"{station_freq_mhz:.4f} MHz"
)

ax.set_xlabel(
    "Фреквенција [MHz]",
    fontsize=14
)

ax.set_ylabel(
    "PSD [dB/Hz]",
    fontsize=14
)

ax.set_title(
    "Поређење прозорских функција при Welch-овој спектралној анализи",
    fontsize=16
)

ax.grid(
    alpha=0.3
)

ax.legend(
    fontsize=10,
    loc="upper right"
)

zoom_x_min = (
    station_freq_mhz
    - 0.012
)

zoom_x_max = (
    station_freq_mhz
    + 0.012
)


zoom_y_values = []


for item in window_curves:

    zoom_mask = (
        (
            item["freqs_mhz"]
            >= zoom_x_min
        )
        &
        (
            item["freqs_mhz"]
            <= zoom_x_max
        )
    )

    zoom_y_values.extend(
        item["psd_db"][
            zoom_mask
        ].tolist()
    )


zoom_y_values = np.asarray(
    zoom_y_values,
    dtype=float
)


if len(zoom_y_values) == 0:

    raise ValueError(
        "Nema tacaka u oblasti izabranoj za zoom."
    )


zoom_y_min = float(
    np.min(
        zoom_y_values
    )
    - 0.4
)

zoom_y_max = float(
    np.max(
        zoom_y_values
    )
    + 0.4
)


axins = create_zoom_axis(
    ax,
    xlim_zoom=(
        zoom_x_min,
        zoom_x_max
    ),
    ylim_zoom=(
        zoom_y_min,
        zoom_y_max
    ),
    bbox_to_anchor=(
        0.06,
        0.06,
        1,
        1
    ),
    width="34%",
    height="34%"
)


for item in window_curves:

    axins.plot(
        item["freqs_mhz"],
        item["psd_db"],
        linewidth=1.3
    )


axins.axvline(
    station_freq_mhz,
    linestyle="--",
    linewidth=1.2
)


mark_inset(
    ax,
    axins,
    loc1=2,
    loc2=4,
    fc="none",
    ec="gray",
    lw=1.0
)


plt.tight_layout()


WINDOW_FIGURE = os.path.join(
    OUTPUT_DIR,
    "03_Welch_prozorske_funkcije.png"
)

plt.savefig(
    WINDOW_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

window_df = pd.DataFrame(
    window_results
)

WINDOW_CSV = os.path.join(
    OUTPUT_DIR,
    "03_Welch_prozorske_funkcije.csv"
)

window_df.to_csv(
    WINDOW_CSV,
    index=False
)


print(
    window_df.to_string(
        index=False
    )
)

print()
print("=" * 90)
print("PARAMETRI EKSPERIMENTA")
print("=" * 90)

print(
    f"FM frekvencija: "
    f"{station_freq_mhz:.6f} MHz"
)

print(
    f"Fs: "
    f"{fs / 1e6:.3f} MS/s"
)

print(
    f"Korisceni segment: "
    f"{SEGMENT_DURATION_SECONDS:.2f} s"
)

print(
    f"Broj I/Q uzoraka: "
    f"{len(x):,}"
)

print(
    f"Broj ponavljanja benchmarka: "
    f"{NUM_REPEATS}"
)

print(
    f"Osnovni Welch nperseg: "
    f"{BASE_NPERSEG}"
)

print(
    f"Osnovni Welch noverlap: "
    f"{BASE_NOVERLAP}"
)

print(
    f"Osnovna rezolucija: "
    f"{fs / BASE_NPERSEG:.2f} Hz"
)


print()
print("=" * 90)
print("SACUVANI FAJLOVI")
print("=" * 90)

for file_path in [
    FFT_WELCH_FIGURE,
    FFT_WELCH_CSV,
    SEGMENT_FIGURE,
    SEGMENT_CSV,
    WINDOW_FIGURE,
    WINDOW_CSV
]:

    print(
        file_path
    )


print()
print("GOTOVO.")