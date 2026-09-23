# -*- coding: utf-8 -*-
import numpy as np


FEATURE_COLUMNS = [
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
    "envelope_modulation_index",
]


def custom_kurtosis(x):
    x = np.asarray(x, dtype=np.float64)
    xc = x - np.mean(x)
    m2 = np.mean(xc ** 2)
    m4 = np.mean(xc ** 4)
    return float(m4 / (m2 ** 2 + 1e-12))


def custom_skewness(x):
    x = np.asarray(x, dtype=np.float64)
    xc = x - np.mean(x)
    m2 = np.mean(xc ** 2)
    m3 = np.mean(xc ** 3)
    return float(m3 / (m2 ** 1.5 + 1e-12))


def extract_features(iq, fs=500_000):
    iq = np.asarray(iq, dtype=np.complex128)

    if iq.ndim != 1 or len(iq) < 256:
        raise ValueError("I/Q segment mora biti 1-D i imati najmanje 256 uzoraka.")
    if not np.all(np.isfinite(iq)):
        raise ValueError("I/Q segment sadrzi NaN ili beskonacne vrednosti.")

    amplitude = np.abs(iq)
    phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(phase)

    amplitude_norm = amplitude / (np.mean(amplitude) + 1e-12)

    amp_var = np.var(amplitude_norm)
    amp_kurtosis = custom_kurtosis(amplitude_norm)
    amp_skew = custom_skewness(amplitude_norm)

    idx = np.arange(len(phase), dtype=np.float64)
    coeff = np.polyfit(idx, phase, 1)
    phase_detrended = phase - np.polyval(coeff, idx)

    phase_var = np.var(phase_detrended)
    phase_kurtosis = custom_kurtosis(phase_detrended)

    freq_var = np.var(inst_freq)
    freq_kurtosis = custom_kurtosis(inst_freq)

    spectrum = np.fft.fftshift(np.fft.fft(iq))
    power = np.abs(spectrum) ** 2
    power_norm = power / (np.sum(power) + 1e-12)

    center = len(power_norm) // 2
    half_width = len(power_norm) // 10
    center_power = np.sum(
        power_norm[center - half_width:center + half_width]
    )

    spectral_flatness = (
        np.exp(np.mean(np.log(power + 1e-12)))
        / (np.mean(power) + 1e-12)
    )

    c20 = np.mean(iq ** 2)
    c21 = np.mean(np.abs(iq) ** 2)
    c40 = np.mean(iq ** 4) - 3.0 * np.mean(iq ** 2) ** 2
    c42 = (
        np.mean(np.abs(iq) ** 4)
        - np.abs(np.mean(iq ** 2)) ** 2
        - 2.0 * np.mean(np.abs(iq) ** 2) ** 2
    )

    n = len(iq)
    window = np.hanning(n)
    spectrum_w = np.fft.fftshift(np.fft.fft(iq * window))
    power_w = np.abs(spectrum_w) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))

    # Procena dominantnog nosioca u centralnih +/- 5 kHz
    carrier_search_mask = np.abs(freqs) <= 5_000.0
    search_indices = np.where(carrier_search_mask)[0]
    local_peak = int(np.argmax(power_w[carrier_search_mask]))
    carrier_idx = int(search_indices[local_peak])
    carrier_freq = float(freqs[carrier_idx])

    # Udeo snage u uskom prozoru +/-500 Hz oko procenjenog nosioca
    carrier_mask = np.abs(freqs - carrier_freq) <= 500.0
    total_mask = np.abs(freqs) <= 100_000.0
    carrier_power_value = float(np.sum(power_w[carrier_mask]))
    total_power_value = float(np.sum(power_w[total_mask]))
    carrier_power_ratio = carrier_power_value / (total_power_value + 1e-12)

    # Simetrija bocnih opsega 0.5-5 kHz levo/desno od nosioca
    left_mask = (
        (freqs >= carrier_freq - 5_000.0)
        & (freqs <= carrier_freq - 500.0)
    )
    right_mask = (
        (freqs >= carrier_freq + 500.0)
        & (freqs <= carrier_freq + 5_000.0)
    )
    left_power = float(np.sum(power_w[left_mask]))
    right_power = float(np.sum(power_w[right_mask]))
    sideband_symmetry = 1.0 - (
        abs(left_power - right_power)
        / (left_power + right_power + 1e-12)
    )
    sideband_symmetry = float(np.clip(sideband_symmetry, 0.0, 1.0))

    # Robustna procena dubine promene amplitudske obvojnice
    envelope = np.abs(iq)
    p05 = float(np.percentile(envelope, 5))
    p95 = float(np.percentile(envelope, 95))
    envelope_modulation_index = (
        (p95 - p05) / (p95 + p05 + 1e-12)
    )
    envelope_modulation_index = float(
        np.clip(envelope_modulation_index, 0.0, 1.0)
    )

    return {
        "amp_var": float(amp_var),
        "amp_kurtosis": float(amp_kurtosis),
        "amp_skew": float(amp_skew),
        "phase_var": float(phase_var),
        "phase_kurtosis": float(phase_kurtosis),
        "freq_var": float(freq_var),
        "freq_kurtosis": float(freq_kurtosis),
        "center_power": float(center_power),
        "spectral_flatness": float(spectral_flatness),
        "C20_abs": float(abs(c20)),
        "C21_abs": float(abs(c21)),
        "C40_abs": float(abs(c40)),
        "C42_abs": float(abs(c42)),
        "carrier_power_ratio": float(carrier_power_ratio),
        "sideband_symmetry": float(sideband_symmetry),
        "envelope_modulation_index": float(envelope_modulation_index),
    }
