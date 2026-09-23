# -*- coding: utf-8 -*-
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

STATION_FREQUENCY_HZ = 98.7047e6
RECEIVER_CENTER_FREQUENCY_HZ = 99.0e6

EXPECTED_SAMPLE_RATE_HZ = 2.4e6

FREQUENCY_OFFSET_HZ = (
    STATION_FREQUENCY_HZ
    - RECEIVER_CENTER_FREQUENCY_HZ
)

WELCH_WINDOW = "hann"
WELCH_NPERSEG = 8192
WELCH_NOVERLAP = 4096
WELCH_NFFT = 8192

CHANNEL_FILTER_TAPS = 257
CHANNEL_LOW_PASS_CUTOFF_HZ = 100e3
CHANNEL_DECIMATION = 10

DEEMPHASIS_TIME_CONSTANT_S = 50e-6

AUDIO_FILTER_TAPS = 129
AUDIO_LOW_PASS_CUTOFF_HZ = 15e3

AUDIO_SAMPLE_RATE_HZ = 48_000

EPS = 1e-20


def clipping_fraction(samples):
    """
    Udeo I/Q uzoraka kod kojih je realna ili imaginarna
    komponenta veoma blizu granici opsega.
    """

    samples = np.asarray(samples)

    return float(
        np.mean(
            (np.abs(samples.real) >= 0.99)
            |
            (np.abs(samples.imag) >= 0.99)
        )
    )


def calculate_welch_psd(iq, fs):
    """
    Welch-ova procena spektralne gustine snage.
    """

    frequencies, psd = signal.welch(
        iq,
        fs=fs,
        window=WELCH_WINDOW,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        nfft=WELCH_NFFT,
        detrend=False,
        return_onesided=False,
        scaling="density"
    )

    frequencies = np.fft.fftshift(
        frequencies
    )

    psd = np.fft.fftshift(
        psd
    )

    psd_db = (
        10.0
        * np.log10(
            np.maximum(
                psd,
                EPS
            )
        )
    )

    return frequencies, psd_db


def frequency_translate(iq, offset_hz, fs):
    """
    Pomera signal sa offset_hz na 0 Hz.
    """

    n = np.arange(
        len(iq),
        dtype=np.float64
    )

    oscillator = np.exp(
        -1j
        * 2.0
        * np.pi
        * offset_hz
        * n
        / fs
    )

    return (
        iq
        * oscillator
    ).astype(
        np.complex64
    )


def estimate_local_background(
    frequencies_hz,
    psd_db
):
    """
    Lokalna spektralna pozadina procenjuje se iz oblasti
    120-150 kHz sa obe strane ciljnog signala.
    """

    noise_mask = (
        (np.abs(frequencies_hz) >= 120e3)
        &
        (np.abs(frequencies_hz) <= 150e3)
    )

    if not np.any(noise_mask):
        raise ValueError(
            "Nema dovoljno tacaka za procenu lokalne pozadine."
        )

    return float(
        np.median(
            psd_db[
                noise_mask
            ]
        )
    )


def fm_demodulate(
    channel_iq,
    fs_channel
):
    """
    FM demodulacija faznim diskriminatorom,
    zatim de-emphasis i audio filtriranje.
    """

    fm_demodulated = np.angle(
        channel_iq[1:]
        *
        np.conj(
            channel_iq[:-1]
        )
    )

    alpha = np.exp(
        -1.0
        /
        (
            fs_channel
            *
            DEEMPHASIS_TIME_CONSTANT_S
        )
    )

    deemphasized = signal.lfilter(
        [
            1.0 - alpha
        ],
        [
            1.0,
            -alpha
        ],
        fm_demodulated
    )


    audio_taps = signal.firwin(
        AUDIO_FILTER_TAPS,
        AUDIO_LOW_PASS_CUTOFF_HZ,
        fs=fs_channel,
        window="hann"
    )

    audio_filtered = signal.lfilter(
        audio_taps,
        [1.0],
        deemphasized
    )

    # Odbacivanje tranzijenta filtra
    audio_filtered = audio_filtered[
        AUDIO_FILTER_TAPS - 1:
    ]

    # 240 kS/s -> 48 kHz
    decimation_factor = int(
        round(
            fs_channel
            /
            AUDIO_SAMPLE_RATE_HZ
        )
    )

    if not np.isclose(
        fs_channel / decimation_factor,
        AUDIO_SAMPLE_RATE_HZ
    ):
        raise ValueError(
            "Audio frekvencija uzorkovanja nije tacno 48 kHz."
        )

    audio = audio_filtered[
        ::decimation_factor
    ]

    # Uklanjanje DC komponente
    audio = (
        audio
        -
        np.mean(audio)
    )

    scale = float(
        np.percentile(
            np.abs(audio),
            99.5
        )
    )

    if scale > 0:
        audio = (
            0.90
            *
            audio
            /
            scale
        )

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    return np.asarray(
        audio,
        dtype=np.float32
    )


def load_recording(file_path):

    data = np.load(
        file_path,
        allow_pickle=True
    )

    if "samples" not in data:
        raise ValueError(
            "NPZ fajl ne sadrzi polje 'samples'."
        )

    iq = data[
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

    print("=" * 80)
    print("FINALNA ANALIZA REALNOG FM SIGNALA")
    print("=" * 80)

    print(
        f"Fajl: {file_path}"
    )

    print(
        f"Broj I/Q uzoraka: {len(iq)}"
    )

    print(
        f"Fs: {fs / 1e6:.3f} MS/s"
    )

    print(
        f"Trajanje: {len(iq) / fs:.3f} s"
    )

    print(
        f"Centar prijemnika: "
        f"{center_freq / 1e6:.6f} MHz"
    )

    print(
        f"Ciljna frekvencija: "
        f"{station_freq / 1e6:.6f} MHz"
    )

    if "gain" in data:
        print(
            f"RF gain: {data['gain']}"
        )

    print()

    if not np.isclose(
        fs,
        EXPECTED_SAMPLE_RATE_HZ
    ):
        raise ValueError(
            f"Ocekivano fs = "
            f"{EXPECTED_SAMPLE_RATE_HZ / 1e6:.1f} MS/s, "
            f"a fajl sadrzi {fs / 1e6:.3f} MS/s."
        )

    if not np.isclose(
        station_freq,
        STATION_FREQUENCY_HZ,
        atol=2e3
    ):
        raise ValueError(
            "Ucitani snimak nije signal oko 98.7047 MHz."
        )

    if not np.isclose(
        center_freq,
        RECEIVER_CENTER_FREQUENCY_HZ,
        atol=2e3
    ):
        raise ValueError(
            "Centar prijemnika nije 99.0 MHz."
        )

    return (
        iq,
        fs,
        center_freq,
        station_freq
    )


def main(
    input_file,
    output_dir
):

    input_file = Path(
        input_file
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        iq,
        fs,
        center_freq,
        station_freq
    ) = load_recording(
        input_file
    )


    iq = (
        iq
        -
        np.mean(iq)
    )

    clipping = clipping_fraction(
        iq
    )

    print(
        f"Udeo odsecanja: "
        f"{100.0 * clipping:.6f}%"
    )


    freq_baseband, psd_db = calculate_welch_psd(
        iq,
        fs
    )

    absolute_frequency = (
        center_freq
        +
        freq_baseband
    )

    # Posmatra se oblast oko ciljne stanice
    target_mask = (
        np.abs(
            absolute_frequency
            -
            station_freq
        )
        <=
        150e3
    )

    target_freq = absolute_frequency[
        target_mask
    ]

    target_psd = psd_db[
        target_mask
    ]

    peak_idx = int(
        np.argmax(
            target_psd
        )
    )

    peak_frequency_hz = float(
        target_freq[
            peak_idx
        ]
    )

    peak_psd_db = float(
        target_psd[
            peak_idx
        ]
    )

    # Pozadina prema ciljnoj frekvenciji
    offsets = (
        target_freq
        -
        station_freq
    )

    background_db = estimate_local_background(
        offsets,
        target_psd
    )

    detection_score_db = (
        peak_psd_db
        -
        background_db
    )

    peak_offset_hz = (
        peak_frequency_hz
        -
        station_freq
    )

    print()
    print("WELCH PSD")
    print("-" * 80)

    print(
        f"Welch: nperseg={WELCH_NPERSEG}, "
        f"noverlap={WELCH_NOVERLAP}, "
        f"nfft={WELCH_NFFT}"
    )

    print(
        f"Frekvencijski razmak: "
        f"{fs / WELCH_NFFT:.2f} Hz"
    )

    print(
        f"Spektralni maksimum: "
        f"{peak_frequency_hz / 1e6:.6f} MHz"
    )

    print(
        f"PSD maksimuma: "
        f"{peak_psd_db:.2f} dB/Hz"
    )

    print(
        f"Lokalna spektralna pozadina: "
        f"{background_db:.2f} dB/Hz"
    )

    print(
        f"Razlika maksimum-pozadina: "
        f"{detection_score_db:.2f} dB"
    )

    print(
        f"Odstupanje maksimuma od cilja: "
        f"{peak_offset_hz:.1f} Hz"
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        target_freq / 1e6,
        target_psd,
        linewidth=0.9
    )

    plt.axvline(
        station_freq / 1e6,
        linestyle="--",
        label="Циљна фреквенција"
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

    plt.savefig(
        output_dir
        /
        "01_fm_98_7047_welch_spectrum.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    freq_offset = (
        station_freq
        -
        center_freq
    )

    print()
    print(
        f"Frekvencijski pomeraj signala: "
        f"{freq_offset / 1e3:.1f} kHz"
    )

    shifted = frequency_translate(
        iq,
        freq_offset,
        fs
    )


    shifted_freq, shifted_psd = (
        calculate_welch_psd(
            shifted,
            fs
        )
    )

    plot_mask = (
        np.abs(
            shifted_freq
        )
        <=
        300e3
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        shifted_freq[
            plot_mask
        ] / 1e3,
        shifted_psd[
            plot_mask
        ],
        linewidth=0.9
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Фреквенција [kHz]"
    )

    plt.ylabel(
        "PSD [dB/Hz]"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "02_fm_shifted_before_filtering.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    channel_taps = signal.firwin(
        CHANNEL_FILTER_TAPS,
        CHANNEL_LOW_PASS_CUTOFF_HZ,
        fs=fs,
        window="hann"
    )

    filtered = signal.lfilter(
        channel_taps,
        [1.0],
        shifted
    )

    # Odbacivanje pocetnog tranzijenta
    filtered_valid = filtered[
        CHANNEL_FILTER_TAPS - 1:
    ]


    filtered_freq, filtered_psd = (
        calculate_welch_psd(
            filtered_valid,
            fs
        )
    )

    plot_mask = (
        np.abs(
            filtered_freq
        )
        <=
        300e3
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        filtered_freq[
            plot_mask
        ] / 1e3,
        filtered_psd[
            plot_mask
        ],
        linewidth=0.9
    )

    plt.xlabel(
        "Фреквенција [kHz]"
    )

    plt.ylabel(
        "PSD [dB/Hz]"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "03_fm_after_rf_filter.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    channel_iq = filtered_valid[
        ::CHANNEL_DECIMATION
    ]

    fs_channel = (
        fs
        /
        CHANNEL_DECIMATION
    )

    print()
    print(
        f"RF FIR: {CHANNEL_FILTER_TAPS} koeficijenata, "
        f"fc={CHANNEL_LOW_PASS_CUTOFF_HZ/1e3:.0f} kHz"
    )

    print(
        f"Posle decimacije: "
        f"{fs_channel / 1e3:.0f} kS/s"
    )

    audio = fm_demodulate(
        channel_iq,
        fs_channel
    )

    audio_duration = (
        len(audio)
        /
        AUDIO_SAMPLE_RATE_HZ
    )

    print()
    print(
        f"Audio fs: "
        f"{AUDIO_SAMPLE_RATE_HZ/1e3:.0f} kHz"
    )

    print(
        f"Trajanje audio signala: "
        f"{audio_duration:.3f} s"
    )

    audio_int16 = np.asarray(
        np.round(
            audio
            *
            32767
        ),
        dtype=np.int16
    )

    wav_path = (
        output_dir
        /
        "fm_98_7047_demodulated.wav"
    )

    wavfile.write(
        wav_path,
        AUDIO_SAMPLE_RATE_HZ,
        audio_int16
    )


    audio_time = (
        np.arange(
            len(audio)
        )
        /
        AUDIO_SAMPLE_RATE_HZ
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        audio_time,
        audio,
        linewidth=0.7
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

    plt.savefig(
        output_dir
        /
        "04_fm_audio_time.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    audio_window = np.hanning(
        len(audio)
    )

    audio_fft = np.fft.rfft(
        audio
        *
        audio_window
    )

    audio_freq = np.fft.rfftfreq(
        len(audio),
        d=1.0 / AUDIO_SAMPLE_RATE_HZ
    )

    audio_power = (
        np.abs(
            audio_fft
        )
        ** 2
    )

    audio_power_db = (
        10
        *
        np.log10(
            audio_power
            +
            EPS
        )
    )

    audio_power_db -= np.max(
        audio_power_db
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        audio_freq / 1e3,
        audio_power_db,
        linewidth=0.8
    )

    plt.xlim(
        0,
        24
    )

    plt.xlabel(
        "Фреквенција [kHz]"
    )

    plt.ylabel(
        "Релативни ниво [dB]"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "05_fm_audio_spectrum.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    plt.figure(
        figsize=(11, 5)
    )

    plt.specgram(
        audio,
        NFFT=1024,
        Fs=AUDIO_SAMPLE_RATE_HZ,
        noverlap=512
    )

    plt.ylim(
        0,
        20_000
    )

    plt.xlabel(
        "Време [s]"
    )

    plt.ylabel(
        "Фреквенција [Hz]"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "06_fm_audio_spectrogram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    summary_path = (
        output_dir
        /
        "fm_98_7047_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FINALNA ANALIZA FM SIGNALA 98.7047 MHz\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            f"Input file: {input_file}\n"
        )

        f.write(
            f"Station frequency: "
            f"{station_freq/1e6:.6f} MHz\n"
        )

        f.write(
            f"Receiver center: "
            f"{center_freq/1e6:.6f} MHz\n"
        )

        f.write(
            f"Sample rate: "
            f"{fs/1e6:.3f} MS/s\n"
        )

        f.write(
            f"IQ duration: "
            f"{len(iq)/fs:.3f} s\n"
        )

        f.write(
            f"Clipping fraction: "
            f"{100*clipping:.6f}%\n"
        )

        f.write(
            f"Welch nperseg: "
            f"{WELCH_NPERSEG}\n"
        )

        f.write(
            f"Welch noverlap: "
            f"{WELCH_NOVERLAP}\n"
        )

        f.write(
            f"Welch nfft: "
            f"{WELCH_NFFT}\n"
        )

        f.write(
            f"Frequency resolution: "
            f"{fs/WELCH_NFFT:.6f} Hz\n"
        )

        f.write(
            f"Detected peak: "
            f"{peak_frequency_hz/1e6:.9f} MHz\n"
        )

        f.write(
            f"Peak PSD: "
            f"{peak_psd_db:.6f} dB/Hz\n"
        )

        f.write(
            f"Local background: "
            f"{background_db:.6f} dB/Hz\n"
        )

        f.write(
            f"Peak-background difference: "
            f"{detection_score_db:.6f} dB\n"
        )

        f.write(
            f"Peak frequency offset: "
            f"{peak_offset_hz:.3f} Hz\n"
        )

        f.write(
            f"Frequency translation: "
            f"{freq_offset/1e3:.3f} kHz\n"
        )

        f.write(
            f"RF FIR taps: "
            f"{CHANNEL_FILTER_TAPS}\n"
        )

        f.write(
            f"RF LPF cutoff: "
            f"{CHANNEL_LOW_PASS_CUTOFF_HZ/1e3:.3f} kHz\n"
        )

        f.write(
            f"RF decimation: "
            f"{CHANNEL_DECIMATION}\n"
        )

        f.write(
            f"Intermediate fs: "
            f"{fs_channel/1e3:.3f} kS/s\n"
        )

        f.write(
            f"De-emphasis: "
            f"{DEEMPHASIS_TIME_CONSTANT_S*1e6:.1f} us\n"
        )

        f.write(
            f"Audio LPF cutoff: "
            f"{AUDIO_LOW_PASS_CUTOFF_HZ/1e3:.3f} kHz\n"
        )

        f.write(
            f"Audio sample rate: "
            f"{AUDIO_SAMPLE_RATE_HZ} Hz\n"
        )

        f.write(
            f"Audio duration: "
            f"{audio_duration:.3f} s\n"
        )

    print()
    print("=" * 80)
    print("ANALIZA ZAVRSENA")
    print("=" * 80)

    print(
        f"WAV: {wav_path.resolve()}"
    )

    print(
        f"Rezime: {summary_path.resolve()}"
    )

    print(
        f"Grafici: {output_dir.resolve()}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Analiza i FM demodulacija "
            "finalnog realnog signala na 98.7047 MHz."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Putanja do FM_98_7047_event_*.npz "
            "fajla generisanog kodom "
            "01_record_real_fm_final.py"
        )
    )

    parser.add_argument(
        "--output",
        default="FM_98_7047_ANALYSIS_RESULTS",
        help="Folder za rezultate."
    )

    args = parser.parse_args()

    main(
        args.input,
        args.output
    )