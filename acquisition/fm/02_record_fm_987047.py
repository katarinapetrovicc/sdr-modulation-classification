import os
import time
from datetime import datetime

import numpy as np
from rtlsdr import RtlSdr


# REALNI FM - FINALNO NEZAVISNO SNIMANJE VISE STANICA

STATIONS = [
    {
        "station_freq": 94.7064e6,
        "center_freq": 95.0e6,
        "name": "94_7064"
    },
    {
        "station_freq": 98.7047e6,
        "center_freq": 99.0e6,
        "name": "98_7047"
    },
    {
        "station_freq": 103.7030e6,
        "center_freq": 104.0e6,
        "name": "103_7030"
    }
]


FS_SDR = 2.4e6
GAIN = "auto"

# Isto trajanje kao kod AM:
PRE_SECONDS = 2.0
POST_SECONDS = 6.0
RECORD_SECONDS = PRE_SECONDS + POST_SECONDS

# 3 nezavisna snimka po stanici
NUM_RECORDINGS = 3

# Razmak izmedju snimaka iste stanice
GAP_SECONDS = 10.0

# Pauza nakon promene frekvencije
TUNE_SETTLE_SECONDS = 1.0


OUTPUT_DIR = "FM_MULTI_2PLUS6_NEW_TEST_20260917"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


sdr = RtlSdr()

sdr.sample_rate = FS_SDR
sdr.gain = GAIN

samples_per_recording = int(
    FS_SDR * RECORD_SECONDS
)


print("=" * 85)
print("REALNI FM - FINALNO NEZAVISNO SNIMANJE VISE STANICA")
print("=" * 85)

print(
    f"Fs SDR: {FS_SDR/1e6:.3f} MS/s"
)

print(
    f"Trajanje jednog snimka: "
    f"{PRE_SECONDS:.1f} s + {POST_SECONDS:.1f} s "
    f"= {RECORD_SECONDS:.1f} s"
)

print(
    f"Broj snimaka po stanici: "
    f"{NUM_RECORDINGS}"
)

print(
    f"Folder: {OUTPUT_DIR}"
)

print()


saved_files = []


try:

    for station_index, station in enumerate(
        STATIONS,
        start=1
    ):

        station_freq = station[
            "station_freq"
        ]

        center_freq = station[
            "center_freq"
        ]

        station_name = station[
            "name"
        ]


        print()
        print("=" * 85)

        print(
            f"STANICA "
            f"{station_index}/{len(STATIONS)}"
        )

        print(
            f"FM frekvencija: "
            f"{station_freq/1e6:.6f} MHz"
        )

        print(
            f"SDR centar: "
            f"{center_freq/1e6:.6f} MHz"
        )

        print("=" * 85)


        # Podesavanje tunera
        sdr.center_freq = center_freq

        time.sleep(
            TUNE_SETTLE_SECONDS
        )


        # Po 3 nezavisna snimka
        for i in range(
            1,
            NUM_RECORDINGS + 1
        ):

            print(
                f"[{i}/{NUM_RECORDINGS}] "
                f"Snimam {RECORD_SECONDS:.1f} s "
                f"(2 s + 6 s)..."
            )


            x = sdr.read_samples(
                samples_per_recording
            ).astype(
                np.complex64
            )


            now = datetime.now()


            filename = os.path.join(
                OUTPUT_DIR,
                "FM_" +
                station_name +
                "_event_" +
                now.strftime(
                    "%Y%m%d_%H%M%S"
                ) +
                ".npz"
            )


            np.savez(
                filename,

                samples=x,

                fs=FS_SDR,

                center_freq=center_freq,

                station_freq=station_freq,

                gain=str(
                    GAIN
                ),

                duration_seconds=RECORD_SECONDS,

                pre_seconds=PRE_SECONDS,

                post_seconds=POST_SECONDS,

                capture_time=str(
                    now
                )
            )


            saved_files.append(
                filename
            )


            print(
                f"SACUVAN: "
                f"{filename}"
            )

            print(
                f"Trajanje IQ zapisa: "
                f"{len(x)/FS_SDR:.2f} s"
            )


            if i < NUM_RECORDINGS:

                print(
                    f"Cekam "
                    f"{GAP_SECONDS:.1f} s "
                    f"do sledeceg snimka..."
                )

                time.sleep(
                    GAP_SECONDS
                )


            print()


finally:

    sdr.close()


print()
print("=" * 85)
print("ZAVRSENO")
print("=" * 85)

print(
    f"Ukupan broj sacuvanih "
    f"FM snimaka: "
    f"{len(saved_files)}"
)

print()

for f in saved_files:

    print(
        f"- {f}"
    )
