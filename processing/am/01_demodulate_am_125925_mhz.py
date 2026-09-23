import numpy as np
from scipy.signal import firwin, lfilter, resample_poly, welch
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import os


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE_NAME = (
    PROJECT_ROOT
    / "data"
    / "am"
    / "AM_125_925"
    / "AM_125925_event_20260917_195720.npz"
)

data = np.load(
    FILE_NAME,
    allow_pickle=True
)

x = data["samples"].astype(
    np.complex128
)

FS = float(
    data["fs"]
)

CENTER_FREQ = float(
    data["center_freq"]
)

TARGET_FREQ = float(
    data["target_freq"]
)


OUTPUT_DIR = os.path.dirname(
    FILE_NAME
)

BASE_NAME = os.path.splitext(
    os.path.basename(
        FILE_NAME
    )
)[0]


print("=" * 75)
print("AM DEMODULACIJA - NOVI NEZAVISNI TEST")
print("=" * 75)

print(
    f"Fajl: {FILE_NAME}"
)

print(
    f"Fs: {FS/1e6:.3f} MS/s"
)

print(
    f"Centar SDR-a: "
    f"{CENTER_FREQ/1e6:.6f} MHz"
)

print(
    f"Ciljna frekvencija: "
    f"{TARGET_FREQ/1e6:.6f} MHz"
)

print(
    f"Trajanje: "
    f"{len(x)/FS:.2f} s"
)

if "trigger_score_db" in data:
    print(
        f"Trigger score: "
        f"{float(data['trigger_score_db']):.2f} dB"
    )

if "detected_peak_freq" in data:
    print(
        f"Detektovani maksimum: "
        f"{float(data['detected_peak_freq'])/1e6:.6f} MHz"
    )

x = (
    x -
    np.mean(x)
)


offset = (
    TARGET_FREQ -
    CENTER_FREQ
)

n = np.arange(
    len(x)
)

bb = (
    x *
    np.exp(
        -1j *
        2 *
        np.pi *
        offset *
        n /
        FS
    )
)


RF_CUTOFF = 6000.0

rf_taps = firwin(
    401,
    RF_CUTOFF,
    fs=FS
)

bb_filt = lfilter(
    rf_taps,
    1.0,
    bb
)

bb_filt = bb_filt[
    401:
]


audio = np.abs(
    bb_filt
)

audio = (
    audio -
    np.mean(audio)
)


audio = resample_poly(
    audio,
    3,
    64
)

FS_AUDIO = 48000

audio_taps = firwin(
    257,
    4000,
    fs=FS_AUDIO
)

audio = lfilter(
    audio_taps,
    1.0,
    audio
)

audio = audio[
    257:
]


mx = np.max(
    np.abs(audio)
)

if mx > 0:
    audio = (
        audio /
        mx
    )

audio = (
    0.9 *
    audio
)

audio_int16 = (
    audio *
    32767
).astype(
    np.int16
)

WAV_FILE = os.path.join(
    OUTPUT_DIR,
    BASE_NAME +
    "_demodulated.wav"
)

write(
    WAV_FILE,
    FS_AUDIO,
    audio_int16
)

print()
print(
    "Sacuvan WAV:"
)

print(
    WAV_FILE
)

print(
    f"Trajanje WAV-a: "
    f"{len(audio)/FS_AUDIO:.2f} s"
)


t = (
    np.arange(
        len(audio)
    ) /
    FS_AUDIO
)

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    t,
    audio
)

plt.xlabel(
    "Време [s]"
)

plt.ylabel(
    "Амплитуда"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

AUDIO_PLOT = os.path.join(
    OUTPUT_DIR,
    BASE_NAME +
    "_audio.png"
)

plt.savefig(
    AUDIO_PLOT,
    dpi=300
)

plt.show()

f, P = welch(
    x,
    fs=FS,
    window="hann",
    nperseg=16384,
    noverlap=8192,
    return_onesided=False,
    scaling="density"
)

f = np.fft.fftshift(
    f
)

P = np.fft.fftshift(
    P
)

rf = (
    CENTER_FREQ +
    f
)

P_db = (
    10 *
    np.log10(
        P + 1e-20
    )
)

mask = (
    (rf >= TARGET_FREQ - 20e3)
    &
    (rf <= TARGET_FREQ + 20e3)
)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    rf[mask] / 1e6,
    P_db[mask]
)

plt.axvline(
    TARGET_FREQ / 1e6,
    linestyle="--",
    label="125.925 MHz"
)

plt.xlabel(
    "Фреквенција [MHz]"
)

plt.ylabel(
    "PSD [dB/Hz]"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

SPECTRUM_PLOT = os.path.join(
    OUTPUT_DIR,
    BASE_NAME +
    "_spectrum.png"
)

plt.savefig(
    SPECTRUM_PLOT,
    dpi=300
)

plt.show()


print()
print("=" * 75)
print("ZAVRSENO")
print("=" * 75)

print(
    "Audio graf:"
)

print(
    AUDIO_PLOT
)

print(
    "RF spektar:"
)

print(
    SPECTRUM_PLOT
)