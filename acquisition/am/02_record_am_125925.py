import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import welch
from collections import deque
from datetime import datetime
import os
import time


# CILJANO SNIMANJE AM KANALA 125.925 MHz
# Cuvamo:
# 2 s PRE detekcije
# 6 s POSLE detekcije
#
# Izlazni folder cuva rezultate


TARGET_FREQ = 125.925e6

# Centar namerno pomeren da cilj ne bude u DC centru
CENTER_FREQ = 125.700e6

FS = 1.024e6


# BLOKOVSKI PRIJEM
BLOCK_DURATION = 0.25
BLOCK_SAMPLES = int(
    FS * BLOCK_DURATION
)


# TRAJANJE SNIMKA DOGADJAJA
PRE_SECONDS = 2.0
POST_SECONDS = 6.0

pre_blocks = int(
    PRE_SECONDS / BLOCK_DURATION
)

post_blocks = int(
    POST_SECONDS / BLOCK_DURATION
)


# UKUPNO VREME SLUSANJA
LISTEN_MINUTES = 20


# PARAMETRI DETEKCIJE
# Posmatramo +/- 10 kHz oko 125.925 MHz
SEARCH_HALF_WIDTH = 10e3

# Sira oblast za procenu lokalne spektralne pozadine
NOISE_HALF_WIDTH = 100e3

THRESHOLD_DB = 8.0

# Da ista transmisija ne napravi vise fajlova
COOLDOWN_SECONDS = 8.0


# WELCH PARAMETRI
NPERSEG = 4096
NOVERLAP = 2048


OUTPUT_DIR = "AM_125_925_NEW_TEST_20260917"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# PRE BUFFER

pre_buffer = deque(
    maxlen=pre_blocks
)

# RTL-SDR
sdr = RtlSdr()

sdr.sample_rate = FS
sdr.center_freq = CENTER_FREQ

# Fiksni gain radi ponovljivosti
sdr.gain = 35.0


print("=" * 80)
print("NOVO CILJANO SNIMANJE AM SIGNALA - NEZAVISNI TEST")
print("=" * 80)

print(
    f"Ciljna frekvencija: "
    f"{TARGET_FREQ/1e6:.6f} MHz"
)

print(
    f"SDR centar: "
    f"{CENTER_FREQ/1e6:.6f} MHz"
)

print(
    f"Fs: "
    f"{FS/1e6:.3f} MS/s"
)

print(
    f"Gain: "
    f"35.0 dB"
)

print(
    f"Welch: "
    f"nperseg={NPERSEG}, "
    f"noverlap={NOVERLAP}"
)

print(
    f"Detekcioni prag: "
    f"{THRESHOLD_DB:.1f} dB"
)

print(
    f"Snima se: "
    f"{PRE_SECONDS:.1f} s PRE + "
    f"{POST_SECONDS:.1f} s POSLE"
)

print(
    f"Ukupno slusanje: "
    f"{LISTEN_MINUTES} min"
)

print(
    f"Folder: "
    f"{OUTPUT_DIR}"
)

print()
print("Cekam transmisiju...")
print()


# FUNKCIJA ZA DETEKCIJU

def calculate_score(x):

    # uklanjanje DC srednje vrednosti
    x2 = (
        x -
        np.mean(x)
    )

    f, pxx = welch(
        x2,
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
        CENTER_FREQ +
        f
    )

    psd_db = (
        10 *
        np.log10(
            pxx + 1e-20
        )
    )



    signal_mask = (
        (rf >= TARGET_FREQ - SEARCH_HALF_WIDTH)
        &
        (rf <= TARGET_FREQ + SEARCH_HALF_WIDTH)
    )


    # OPSEG ZA PROCENU POZADINE

    noise_mask = (
        (rf >= TARGET_FREQ - NOISE_HALF_WIDTH)
        &
        (rf <= TARGET_FREQ + NOISE_HALF_WIDTH)
        &
        ~signal_mask
    )


    signal_psd = psd_db[
        signal_mask
    ]

    signal_rf = rf[
        signal_mask
    ]


    # Najjaci pik unutar +/- 10 kHz oko ciljne frekvencije
    peak_index = np.argmax(
        signal_psd
    )

    signal_level = signal_psd[
        peak_index
    ]

    peak_freq = signal_rf[
        peak_index
    ]


    noise_level = np.median(
        psd_db[
            noise_mask
        ]
    )


    score = (
        signal_level -
        noise_level
    )


    return (
        score,
        signal_level,
        noise_level,
        peak_freq
    )


start_time = time.time()

last_trigger_time = 0.0

event_number = 0


try:

    while (
        time.time() - start_time
        <
        LISTEN_MINUTES * 60
    ):

        x = sdr.read_samples(
            BLOCK_SAMPLES
        )

        now = datetime.now()

        (
            score,
            signal_level,
            noise_level,
            peak_freq
        ) = calculate_score(x)


        print(
            now.strftime("%H:%M:%S"),
            "|",
            f"score = {score:6.2f} dB",
            "|",
            f"peak = {peak_freq/1e6:.6f} MHz"
        )


        # ----------------------------------------------------
        # PROVERA PRAGA
        # ----------------------------------------------------

        triggered = (
            score >= THRESHOLD_DB
            and
            (
                time.time() -
                last_trigger_time
                >
                COOLDOWN_SECONDS
            )
        )


        if triggered:

            event_number += 1

            trigger_time = now


            print()
            print("!" * 80)

            print(
                f"DETEKTOVANA TRANSMISIJA #{event_number}"
            )

            print(
                f"Vreme: "
                f"{trigger_time}"
            )

            print(
                f"Score: "
                f"{score:.2f} dB"
            )

            print(
                f"Signal level: "
                f"{signal_level:.2f} dB/Hz"
            )

            print(
                f"Noise level: "
                f"{noise_level:.2f} dB/Hz"
            )

            print(
                f"Detektovani maksimum: "
                f"{peak_freq/1e6:.6f} MHz"
            )

            print(
                f"Cuvam "
                f"{PRE_SECONDS:.1f} s PRE + "
                f"{POST_SECONDS:.1f} s POSLE..."
            )

            print("!" * 80)
            print()


            # PRE BLOKOVI
            event_blocks = list(
                pre_buffer
            )


            # trenutni blok
            event_blocks.append(
                x.copy()
            )


            # POST BLOKOVI
            for k in range(
                post_blocks
            ):

                post_x = sdr.read_samples(
                    BLOCK_SAMPLES
                )

                event_blocks.append(
                    post_x.copy()
                )


            iq_event = np.concatenate(
                event_blocks
            ).astype(
                np.complex64
            )


            duration = (
                len(iq_event) /
                FS
            )


            filename = os.path.join(
                OUTPUT_DIR,
                "AM_125925_event_" +
                trigger_time.strftime(
                    "%Y%m%d_%H%M%S"
                ) +
                ".npz"
            )


            np.savez(
                filename,

                samples=iq_event,

                fs=FS,

                center_freq=CENTER_FREQ,

                target_freq=TARGET_FREQ,

                detected_peak_freq=peak_freq,

                trigger_time=str(
                    trigger_time
                ),

                trigger_score_db=score,

                signal_level_db=signal_level,

                noise_level_db=noise_level,

                duration_seconds=duration,

                pre_seconds=PRE_SECONDS,

                post_seconds=POST_SECONDS,

                gain_db=35.0,

                nperseg=NPERSEG,

                noverlap=NOVERLAP,

                search_half_width_hz=SEARCH_HALF_WIDTH,

                noise_half_width_hz=NOISE_HALF_WIDTH,

                threshold_db=THRESHOLD_DB
            )


            print(
                "SACUVAN:"
            )

            print(
                filename
            )

            print(
                f"Trajanje IQ zapisa: "
                f"{duration:.2f} s"
            )

            print()


            last_trigger_time = time.time()


            # reset PRE bafera
            pre_buffer.clear()


        else:

            pre_buffer.append(
                x.copy()
            )


finally:

    sdr.close()

print()
print("=" * 80)
print("ZAVRSENO")
print("=" * 80)

print(
    f"Broj sacuvanih transmisija: "
    f"{event_number}"
)

print(
    "Rezultati su u folderu:"
)

print(
    OUTPUT_DIR
)