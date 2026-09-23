import numpy as np
import matplotlib.pyplot as plt

from rtlsdr import RtlSdr
from scipy.signal import welch

from datetime import datetime

import os
import time
import csv


# SKENIRANJE VHF AIRBAND OPSEGA
# 118 - 137 MHz
#
# Cilj:
# pronalazenje aktivnih vazduhoplovnih AM kanala
#
# Rezultati se cuvaju u folderu


START_FREQ = 118.0e6
STOP_FREQ = 137.0e6

FS = 1.024e6

# Deo spektra koji realno koristimo oko svakog centra
# Izbegavamo krajeve prijemnog opsega
USABLE_HALF_WIDTH = 400e3

# Korak izmedju centralnih frekvencija
# Opsezi se blago preklapaju
STEP = 700e3

# Koliko dugo se prima na jednom centru
DWELL_SECONDS = 0.30

# Ukupno vreme skeniranja
SCAN_MINUTES = 10

# Fiksno pojacanje radi ponovljivosti
GAIN_DB = 35.0


# WELCH PARAMETRI
NPERSEG = 4096
NOVERLAP = 2048


# DETEKCIJA
# Kandidat mora biti ovoliko dB iznad lokalne pozadine.
DETECTION_THRESHOLD_DB = 6.0

# Kandidati iznad ove vrednosti se posebno cuvaju.
SAVE_THRESHOLD_DB = 7.0

# DC oblast oko centra koju ignorisemo.
DC_EXCLUDE_HZ = 30e3


# OUTPUT FOLDER
OUTPUT_DIR = "AM_AIRBAND_SCAN_NEW_TEST_20260917"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# CENTRALNE FREKVENCIJE
centers = np.arange(
    START_FREQ + USABLE_HALF_WIDTH,
    STOP_FREQ - USABLE_HALF_WIDTH + STEP,
    STEP
)

centers = centers[
    centers <= STOP_FREQ - USABLE_HALF_WIDTH
]


# RTL-SDR
sdr = RtlSdr()

sdr.sample_rate = FS
sdr.gain = GAIN_DB



detections = []

scan_number = 0

start_time = time.time()


print("=" * 80)
print("PONOVLJENO SKENIRANJE VHF AIRBAND OPSEGA")
print("=" * 80)

print(
    f"Opseg: "
    f"{START_FREQ/1e6:.3f} - "
    f"{STOP_FREQ/1e6:.3f} MHz"
)

print(
    f"Fs: {FS/1e6:.3f} MS/s"
)

print(
    f"Gain: {GAIN_DB:.1f} dB"
)

print(
    f"Korak skeniranja: "
    f"{STEP/1e3:.0f} kHz"
)

print(
    f"Zadrzavanje po centru: "
    f"{DWELL_SECONDS:.2f} s"
)

print(
    f"Welch: "
    f"nperseg={NPERSEG}, "
    f"noverlap={NOVERLAP}"
)

print(
    f"Prag detekcije: "
    f"{DETECTION_THRESHOLD_DB:.1f} dB"
)

print(
    f"Trajanje: "
    f"{SCAN_MINUTES} min"
)

print()

print(
    "Posebno pratimo da li se ponovo pojavljuje "
    "aktivnost oko 125.925 MHz."
)

print()


# GLAVNO SKENIRANJE
try:

    while (
        time.time() - start_time
        <
        SCAN_MINUTES * 60
    ):

        scan_number += 1

        print()
        print("-" * 80)

        print(
            f"CIKLUS SKENIRANJA #{scan_number}"
        )

        print("-" * 80)


        for center_freq in centers:

            # ------------------------------------------------
            # PODESI SDR
            # ------------------------------------------------

            sdr.center_freq = center_freq

            # kratko cekanje da se tuner stabilizuje
            time.sleep(0.05)

            number_of_samples = int(
                FS * DWELL_SECONDS
            )

            x = sdr.read_samples(
                number_of_samples
            )

            now = datetime.now()


            # ------------------------------------------------
            # UKLANJANJE DC SREDNJE VREDNOSTI
            # ------------------------------------------------

            x = (
                x -
                np.mean(x)
            )


            # ------------------------------------------------
            # WELCH PSD
            # ------------------------------------------------

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

            rf = (
                center_freq +
                f
            )

            psd_db = (
                10 *
                np.log10(
                    pxx + 1e-20
                )
            )

            valid_mask = (
                (rf >= center_freq - USABLE_HALF_WIDTH)
                &
                (rf <= center_freq + USABLE_HALF_WIDTH)
                &
                (
                    np.abs(
                        rf - center_freq
                    )
                    >=
                    DC_EXCLUDE_HZ
                )
                &
                (rf >= START_FREQ)
                &
                (rf <= STOP_FREQ)
            )


            valid_rf = rf[
                valid_mask
            ]

            valid_psd = psd_db[
                valid_mask
            ]


            if len(valid_psd) == 0:
                continue


            # POZADINA SPEKTRA

            noise_level = np.median(
                valid_psd
            )


            # NAJJACI PIK U OVOM SEGMENTU

            peak_index = np.argmax(
                valid_psd
            )

            peak_freq = valid_rf[
                peak_index
            ]

            peak_level = valid_psd[
                peak_index
            ]

            score = (
                peak_level -
                noise_level
            )


            print(
                now.strftime("%H:%M:%S"),
                "| centar:",
                f"{center_freq/1e6:9.3f} MHz",
                "| pik:",
                f"{peak_freq/1e6:10.6f} MHz",
                "|",
                f"{score:6.2f} dB"
            )


            # DETEKCIJA
            if (
                score
                >=
                DETECTION_THRESHOLD_DB
            ):

                result = {
                    "time": now,
                    "center_freq": center_freq,
                    "peak_freq": peak_freq,
                    "peak_level": peak_level,
                    "noise_level": noise_level,
                    "score": score
                }

                detections.append(
                    result
                )


                if (
                    score
                    >=
                    SAVE_THRESHOLD_DB
                ):

                    print(
                        "   >>> JAK KANDIDAT:",
                        f"{peak_freq/1e6:.6f} MHz",
                        f"| {score:.2f} dB <<<"
                    )


                    filename = os.path.join(
                        OUTPUT_DIR,
                        "candidate_" +
                        now.strftime(
                            "%Y%m%d_%H%M%S"
                        ) +
                        "_" +
                        f"{peak_freq/1e6:.6f}".replace(
                            ".",
                            "_"
                        ) +
                        "MHz.npz"
                    )


                    np.savez(
                        filename,
                        samples=x.astype(
                            np.complex64
                        ),
                        fs=FS,
                        center_freq=center_freq,
                        detected_peak_freq=peak_freq,
                        peak_level_db=peak_level,
                        noise_level_db=noise_level,
                        score_db=score,
                        timestamp=str(now),
                        gain_db=GAIN_DB,
                        nperseg=NPERSEG,
                        noverlap=NOVERLAP
                    )


finally:

    sdr.close()



detections_sorted = sorted(
    detections,
    key=lambda d: d["score"],
    reverse=True
)


csv_file = os.path.join(
    OUTPUT_DIR,
    "airband_detections.csv"
)

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as f_csv:

    writer = csv.writer(
        f_csv
    )

    writer.writerow([
        "time",
        "center_freq_MHz",
        "peak_freq_MHz",
        "peak_level_dB_Hz",
        "noise_level_dB_Hz",
        "score_dB"
    ])

    for d in detections_sorted:

        writer.writerow([
            str(d["time"]),
            d["center_freq"] / 1e6,
            d["peak_freq"] / 1e6,
            d["peak_level"],
            d["noise_level"],
            d["score"]
        ])

txt_file = os.path.join(
    OUTPUT_DIR,
    "airband_top_candidates.txt"
)

with open(
    txt_file,
    "w",
    encoding="utf-8"
) as f_txt:

    f_txt.write(
        "NAJJACI AIRBAND KANDIDATI\n"
    )

    f_txt.write(
        "=" * 80 +
        "\n"
    )

    for i, d in enumerate(
        detections_sorted[:30],
        start=1
    ):

        line = (
            f"{i:2d}. "
            f"{d['time'].strftime('%H:%M:%S')} | "
            f"{d['peak_freq']/1e6:.6f} MHz | "
            f"{d['score']:.2f} dB\n"
        )

        f_txt.write(
            line
        )


# GRAFIK NAJJACIH DETEKCIJA
if len(
    detections_sorted
) > 0:

    top = detections_sorted[
        :min(
            30,
            len(detections_sorted)
        )
    ]

    frequencies = np.array([
        d["peak_freq"] / 1e6
        for d in top
    ])

    scores = np.array([
        d["score"]
        for d in top
    ])


    plt.figure(
        figsize=(12, 6)
    )

    plt.scatter(
        frequencies,
        scores
    )

    plt.axhline(
        DETECTION_THRESHOLD_DB,
        linestyle="--",
        label="Prag detekcije"
    )

    # oznacimo poznatu frekvenciju koju zelimo da proverimo
    plt.axvline(
        125.925,
        linestyle="--",
        label="125.925 MHz"
    )

    plt.xlabel(
        "Frekvencija [MHz]"
    )

    plt.ylabel(
        "Signal iznad pozadine [dB]"
    )

    plt.title(
        "Najizrazeniji kandidati u VHF airband opsegu"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "airband_top_candidates.png"
        ),
        dpi=300
    )

    plt.show()

print()
print("=" * 80)
print("ZAVRSENO SKENIRANJE")
print("=" * 80)

print(
    f"Ukupan broj detekcija: "
    f"{len(detections_sorted)}"
)

print()

print(
    "30 NAJJACIH KANDIDATA:"
)

print("-" * 80)


for i, d in enumerate(
    detections_sorted[:30],
    start=1
):

    print(
        f"{i:2d}. "
        f"{d['time'].strftime('%H:%M:%S')} | "
        f"{d['peak_freq']/1e6:.6f} MHz | "
        f"{d['score']:.2f} dB"
    )


print()
print(
    "Rezultati sacuvani u:"
)

print(
    OUTPUT_DIR
)