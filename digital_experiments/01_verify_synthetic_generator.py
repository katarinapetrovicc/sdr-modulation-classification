# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

SEED = 42
rng = np.random.default_rng(SEED)

SYMBOL_RATE = 100_000
SPS_SOURCE = 4
FS_SOURCE = SYMBOL_RATE * SPS_SOURCE
FS_FINAL = 2_000_000
UPSAMPLE_FACTOR = 5
RRC_ALPHA = 0.35
RRC_SPAN_SYMBOLS = 10
N_SYMBOLS = 5000

OUTPUT_DIR = "VERIFIKACIJA_SINTETICKOG_GENERATORA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"


def rrc_taps(beta, sps, span_symbols=10):
    n_taps = span_symbols * sps + 1
    t = (np.arange(n_taps) - (n_taps - 1) / 2) / sps
    h = np.zeros(n_taps, dtype=float)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 + beta * (4 / np.pi - 1)
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-10:
            h[i] = (
                beta / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den

    h /= np.sqrt(np.sum(h**2) + 1e-12)
    return h


RRC = rrc_taps(RRC_ALPHA, SPS_SOURCE, RRC_SPAN_SYMBOLS)


def gen_bpsk(n):
    bits = rng.integers(0, 2, n)
    return (2 * bits - 1).astype(np.complex64)


def gen_qpsk(n):
    idx = rng.integers(0, 4, n)
    return np.exp(1j * (np.pi / 4 + idx * np.pi / 2)).astype(np.complex64)


def gen_16qam(n):
    lev = np.array([-3, -1, 1, 3], dtype=float)
    i = rng.choice(lev, n)
    q = rng.choice(lev, n)
    return ((i + 1j * q) / np.sqrt(10)).astype(np.complex64)


def waveform(symbols):
    x = signal.upfirdn(RRC, symbols, up=SPS_SOURCE)
    d = (len(RRC) - 1) // 2
    x = x[d:len(x) - d]
    y = signal.resample_poly(x, up=UPSAMPLE_FACTOR, down=1).astype(np.complex64)
    p = np.mean(np.abs(y) ** 2)
    return (y / np.sqrt(p + 1e-12)).astype(np.complex64)


def bw99(iq):
    spec = np.fft.fftshift(np.fft.fft(iq))
    power = np.abs(spec) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1 / FS_FINAL))
    cdf = np.cumsum(power) / (np.sum(power) + 1e-12)

    lo = min(np.searchsorted(cdf, 0.005), len(freq) - 1)
    hi = min(np.searchsorted(cdf, 0.995), len(freq) - 1)

    return float(freq[hi] - freq[lo])


def save_constellation(symbols, label):
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(symbols), np.imag(symbols), s=12, alpha=0.45)
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("I компонента")
    plt.ylabel("Q компонента")
    plt.title(f"{label} – констелациони дијаграм")
    plt.grid(True, alpha=0.25)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{label}_konstelacija.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def save_psd(iq, label):
    f, pxx = signal.welch(
        iq,
        fs=FS_FINAL,
        window="hann",
        nperseg=4096,
        noverlap=2048,
        return_onesided=False,
        scaling="density"
    )
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)

    plt.figure(figsize=(8, 5))
    plt.plot(f / 1e3, 10 * np.log10(pxx + 1e-20))
    plt.xlabel("Фреквенција [kHz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.title(f"{label} – Welch-ова PSD")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{label}_PSD.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def save_amp(iq, label):
    plt.figure(figsize=(7, 5))
    plt.hist(np.abs(iq), bins=80, density=True, alpha=0.8)
    plt.xlabel("Амплитуда |x[n]|")
    plt.ylabel("Густина")
    plt.title(f"{label} – расподела амплитуде")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{label}_amplitudska_raspodela.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


gens = {
    "BPSK": gen_bpsk,
    "QPSK": gen_qpsk,
    "16-QAM": gen_16qam
}

rows = []

print("=" * 90)
print("ВЕРИФИКАЦИЈА СИНТЕТИЧКОГ ГЕНЕРАТОРА")
print("=" * 90)
print("100 kSym/s -> 4 sps -> 400 kS/s -> RRC 0.35 -> x5 -> 2 MS/s")

for label, fn in gens.items():
    print("Обрађујем:", label)

    symbols = fn(N_SYMBOLS)
    iq = waveform(symbols)

    save_constellation(symbols, label)
    save_psd(iq, label)
    save_amp(iq, label)

    rows.append({
        "modulacija": label,
        "broj_simbola": len(symbols),
        "symbol_rate_kSym_s": SYMBOL_RATE / 1e3,
        "source_sps": SPS_SOURCE,
        "source_fs_kS_s": FS_SOURCE / 1e3,
        "final_fs_MS_s": FS_FINAL / 1e6,
        "rrc_alpha": RRC_ALPHA,
        "srednja_snaga": float(np.mean(np.abs(iq) ** 2)),
        "srednja_amplituda": float(np.mean(np.abs(iq))),
        "std_amplitude": float(np.std(np.abs(iq))),
        "PAPR_dB": float(
            10 * np.log10(np.max(np.abs(iq) ** 2) / (np.mean(np.abs(iq) ** 2) + 1e-12))
        ),
        "BW_99_kHz": bw99(iq) / 1e3,
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTPUT_DIR, "generator_verification_stats.csv"), index=False)

print("\nНУМЕРИЧКЕ СТАТИСТИКЕ")
print(df.to_string(index=False))

print("\nСачувано у:", OUTPUT_DIR)
print("- BPSK/QPSK/16-QAM констелације")
print("- BPSK/QPSK/16-QAM Welch PSD")
print("- BPSK/QPSK/16-QAM расподеле амплитуде")
print("- generator_verification_stats.csv")