"""Build reproducible, fixed-window mHealth features for M3-Impute.

Windows are placed on the raw time axis without consulting activity labels.
The default supervised table retains only windows wholly labeled with one of
the 12 activities. Every candidate window and its selection decision is
recorded in windows.csv. Feature extraction itself needs no activity labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "mhealth" / "raw_data"
SUBJECTS = range(1, 11)
SAMPLE_RATE = 50.0
WINDOW_SIZE = 128
STRIDE = 64
AXES = "xyz"
GRAVITY = 9.80665
MODE_NAMES = ("full", "acc_gyro_only")
FREQUENCY_BANDS = ((0.3, 3.0), (3.0, 8.0), (8.0, 20.0))


def fir_lowpass(cutoff_hz: float, taps: int) -> np.ndarray:
    """Hamming-windowed sinc FIR with unit DC gain."""
    if taps < 3 or taps % 2 != 1 or not 0 < cutoff_hz < SAMPLE_RATE / 2:
        raise ValueError("FIR requires odd taps >= 3 and a cutoff below Nyquist")
    offsets = np.arange(taps) - taps // 2
    kernel = 2 * cutoff_hz / SAMPLE_RATE * np.sinc(2 * cutoff_hz / SAMPLE_RATE * offsets)
    kernel *= np.hamming(taps)
    return kernel / kernel.sum()


def filter_continuous(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Zero-phase linear FIR convolution on a whole subject, with reflection."""
    padding = len(kernel) // 2
    padded = np.pad(values, ((padding, padding), (0, 0)), mode="reflect")
    fft_size = 1 << (len(padded) + len(kernel) - 2).bit_length()
    signal_fft = np.fft.rfft(padded, n=fft_size, axis=0)
    kernel_fft = np.fft.rfft(kernel, n=fft_size)
    convolution = np.fft.irfft(signal_fft * kernel_fft[:, None], n=fft_size, axis=0)
    centered = convolution[padding:padding + len(padded)]
    return centered[padding:-padding]


def load_subject(path: Path) -> np.ndarray:
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] != 24 or len(raw) < WINDOW_SIZE:
        raise ValueError(f"{path}: expected at least {WINDOW_SIZE} rows and 24 columns")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: non-finite sensor value or label")
    labels = raw[:, -1]
    if np.any(labels != np.floor(labels)) or np.any((labels < 0) | (labels > 12)):
        raise ValueError(f"{path}: labels must be integers from 0 through 12")
    return raw


def prepare_signals(raw: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    """Convert units and derive motion signals before window extraction."""
    if mode not in MODE_NAMES:
        raise ValueError(f"unknown mode: {mode}")
    denoise = fir_lowpass(20.0, 101)
    gravity_filter = fir_lowpass(0.3, 501)
    signals = {}
    for place, columns in (("chest", slice(0, 3)), ("ankle", slice(5, 8)),
                           ("wrist", slice(14, 17))):
        acceleration = filter_continuous(raw[:, columns] / GRAVITY, denoise)
        gravity = filter_continuous(acceleration, gravity_filter)
        body = acceleration - gravity
        signals[f"body_acc_{place}"] = body
        signals[f"gravity_acc_{place}"] = gravity
        signals[f"body_acc_jerk_{place}"] = np.gradient(body, 1 / SAMPLE_RATE, axis=0)

    for place, columns in (("ankle", slice(8, 11)), ("wrist", slice(17, 20))):
        gyro = filter_continuous(np.deg2rad(raw[:, columns]), denoise)
        signals[f"gyro_{place}"] = gyro
        signals[f"gyro_jerk_{place}"] = np.gradient(gyro, 1 / SAMPLE_RATE, axis=0)

    if mode == "full":
        for place, columns in (("ankle", slice(11, 14)), ("wrist", slice(20, 23))):
            signals[f"mag_{place}"] = raw[:, columns]
        signals["ecg"] = raw[:, 3:5]
    return signals


def label_windows(labels: np.ndarray, window_size: int, stride: int,
                  min_purity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign majority labels after fixing all window positions."""
    if window_size < 2 or stride < 1 or not 0 < min_purity <= 1:
        raise ValueError("invalid window size, stride, or minimum purity")
    starts = np.arange(0, len(labels) - window_size + 1, stride, dtype=int)
    if not len(starts):
        raise ValueError("subject is shorter than one window")
    window_labels = labels[starts[:, None] + np.arange(window_size)]
    counts = np.stack([(window_labels == label).sum(axis=1) for label in range(13)], axis=1)
    majority = counts.argmax(axis=1)
    purity = counts[np.arange(len(starts)), majority] / window_size
    selected = (majority != 0) & (purity >= min_purity)
    return starts, majority, purity, selected


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator > 1e-12)


def _correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    centered_a = a - a.mean(axis=1, keepdims=True)
    centered_b = b - b.mean(axis=1, keepdims=True)
    covariance = np.mean(centered_a * centered_b, axis=1)
    return np.clip(_ratio(covariance, a.std(axis=1) * b.std(axis=1)), -1, 1)


def _spectral_features(values: np.ndarray) -> dict[str, np.ndarray]:
    """Windowed periodogram summaries; bands are fractions of non-DC power."""
    centered = values - values.mean(axis=1, keepdims=True)
    taper = np.hanning(values.shape[1])
    power = np.abs(np.fft.rfft(centered * taper[None, :], axis=1)) ** 2
    frequencies = np.fft.rfftfreq(values.shape[1], 1 / SAMPLE_RATE)
    power[:, 0] = 0
    total = power.sum(axis=1)
    probabilities = _ratio(power, total[:, None])
    result = {
        "dominant_hz": frequencies[power.argmax(axis=1)],
        "spectral_centroid_hz": _ratio((power * frequencies).sum(axis=1), total),
        "spectral_entropy": -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1),
    }
    for low, high in FREQUENCY_BANDS:
        band = (frequencies >= low) & (frequencies < high)
        result[f"power_{low:g}_{high:g}_fraction"] = _ratio(power[:, band].sum(axis=1), total)
    return result


def extract_features(signals: dict[str, np.ndarray], starts: np.ndarray,
                     window_size: int) -> tuple[np.ndarray, list[str]]:
    """Extract named, physically grouped features from fixed windows."""
    columns: list[np.ndarray] = []
    names: list[str] = []
    offsets = starts[:, None] + np.arange(window_size)

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        columns.append(np.asarray(values, dtype=float))

    def windows(key: str) -> np.ndarray:
        return signals[key][offsets]

    def primary(key: str) -> None:
        values = windows(key)
        for axis, name in enumerate(AXES):
            channel = values[:, :, axis]
            median = np.median(channel, axis=1)
            add(f"{key}_{name}_mean", channel.mean(axis=1))
            add(f"{key}_{name}_std", channel.std(axis=1))
            add(f"{key}_{name}_mad", np.median(np.abs(channel - median[:, None]), axis=1))
            add(f"{key}_{name}_iqr", np.quantile(channel, .75, axis=1) - np.quantile(channel, .25, axis=1))
            add(f"{key}_{name}_energy", np.mean(channel ** 2, axis=1))
            add(f"{key}_{name}_lag1_corr", _correlation(channel[:, :-1], channel[:, 1:]))
        magnitude = np.linalg.norm(values, axis=2)
        add(f"{key}_mag_mean", magnitude.mean(axis=1))
        add(f"{key}_mag_std", magnitude.std(axis=1))
        add(f"{key}_mag_energy", np.mean(magnitude ** 2, axis=1))
        add(f"{key}_sma", np.mean(np.abs(values).sum(axis=2), axis=1))
        for statistic, result in _spectral_features(magnitude).items():
            add(f"{key}_mag_{statistic}", result)

    def gravity(key: str) -> None:
        values = windows(key)
        for axis, name in enumerate(AXES):
            add(f"{key}_{name}_mean", values[:, :, axis].mean(axis=1))
            add(f"{key}_{name}_std", values[:, :, axis].std(axis=1))
        magnitude = np.linalg.norm(values, axis=2)
        add(f"{key}_mag_mean", magnitude.mean(axis=1))
        add(f"{key}_mag_std", magnitude.std(axis=1))
        mean_vector = values.mean(axis=1)
        length = np.linalg.norm(mean_vector, axis=1)
        for axis, name in enumerate(AXES):
            cosine = np.clip(_ratio(mean_vector[:, axis], length), -1, 1)
            add(f"{key}_angle_{name}_rad", np.arccos(cosine))

    def jerk(key: str) -> None:
        values = windows(key)
        for axis, name in enumerate(AXES):
            channel = values[:, :, axis]
            add(f"{key}_{name}_rms", np.sqrt(np.mean(channel ** 2, axis=1)))
            add(f"{key}_{name}_iqr", np.quantile(channel, .75, axis=1) - np.quantile(channel, .25, axis=1))
        magnitude = np.linalg.norm(values, axis=2)
        add(f"{key}_mag_rms", np.sqrt(np.mean(magnitude ** 2, axis=1)))
        spectral = _spectral_features(magnitude)
        for statistic in ("dominant_hz", "spectral_entropy",
                          "power_0.3_3_fraction", "power_3_8_fraction", "power_8_20_fraction"):
            add(f"{key}_mag_{statistic}", spectral[statistic])

    def magnetometer(key: str) -> None:
        values = windows(key)
        strength = np.linalg.norm(values, axis=2)
        direction = _ratio(values, strength[:, :, None])
        for axis, name in enumerate(AXES):
            add(f"{key}_direction_{name}_mean", direction[:, :, axis].mean(axis=1))
            add(f"{key}_direction_{name}_std", direction[:, :, axis].std(axis=1))
        add(f"{key}_direction_concentration", np.linalg.norm(direction.mean(axis=1), axis=1))
        log_strength = np.log1p(strength)
        add(f"{key}_log_strength_median", np.median(log_strength, axis=1))
        add(f"{key}_log_strength_iqr", np.quantile(log_strength, .75, axis=1)
            - np.quantile(log_strength, .25, axis=1))
        add(f"{key}_log_strength_std", log_strength.std(axis=1))
        direction_change = np.linalg.norm(np.diff(direction, axis=1), axis=2)
        add(f"{key}_direction_change_mean", direction_change.mean(axis=1))
        add(f"{key}_direction_change_std", direction_change.std(axis=1))
        spectral = _spectral_features(log_strength)
        for statistic in ("dominant_hz", "spectral_entropy",
                          "power_0.3_3_fraction", "power_3_8_fraction", "power_8_20_fraction"):
            add(f"{key}_log_strength_{statistic}", spectral[statistic])

    for place in ("chest", "ankle", "wrist"):
        primary(f"body_acc_{place}")
    for place in ("ankle", "wrist"):
        primary(f"gyro_{place}")
    for place in ("chest", "ankle", "wrist"):
        gravity(f"gravity_acc_{place}")
        jerk(f"body_acc_jerk_{place}")
    for place in ("ankle", "wrist"):
        jerk(f"gyro_jerk_{place}")

    for place in ("chest", "ankle", "wrist"):
        values = windows(f"body_acc_{place}")
        for first, second in ((0, 1), (0, 2), (1, 2)):
            add(f"body_acc_{place}_corr_{AXES[first]}{AXES[second]}",
                _correlation(values[:, :, first], values[:, :, second]))
    for place in ("ankle", "wrist"):
        values = windows(f"gyro_{place}")
        for first, second in ((0, 1), (0, 2), (1, 2)):
            add(f"gyro_{place}_corr_{AXES[first]}{AXES[second]}",
                _correlation(values[:, :, first], values[:, :, second]))
        add(f"acc_gyro_{place}_mag_corr",
            _correlation(np.linalg.norm(windows(f"body_acc_{place}"), axis=2),
                         np.linalg.norm(values, axis=2)))
    magnitudes = {place: np.linalg.norm(windows(f"body_acc_{place}"), axis=2)
                  for place in ("chest", "ankle", "wrist")}
    for first, second in (("chest", "ankle"), ("chest", "wrist"), ("ankle", "wrist")):
        add(f"body_acc_mag_corr_{first}_{second}", _correlation(magnitudes[first], magnitudes[second]))

    if "ecg" in signals:
        for place in ("ankle", "wrist"):
            magnetometer(f"mag_{place}")
        ecg = windows("ecg")
        for lead in range(2):
            channel = ecg[:, :, lead]
            add(f"ecg_{lead + 1}_mean", channel.mean(axis=1))
            add(f"ecg_{lead + 1}_std", channel.std(axis=1))
            add(f"ecg_{lead + 1}_iqr", np.quantile(channel, .75, axis=1) - np.quantile(channel, .25, axis=1))
            add(f"ecg_{lead + 1}_energy", np.mean(channel ** 2, axis=1))

    features = np.column_stack(columns)
    expected = 295 if "ecg" in signals else 253
    if features.shape != (len(starts), expected) or len(names) != len(set(names)):
        raise ValueError("unexpected feature schema")
    if not np.isfinite(features).all():
        raise ValueError("feature extraction produced non-finite values")
    return features, names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source_dir: Path, output_dir: Path, mode: str, window_size: int,
            stride: int, min_purity: float) -> None:
    if mode not in MODE_NAMES:
        raise ValueError(f"mode must be one of {MODE_NAMES}")
    if window_size != WINDOW_SIZE:
        raise ValueError(f"feature schema requires {WINDOW_SIZE}-sample windows")
    if stride < 1 or not 0 < min_purity <= 1:
        raise ValueError("stride must be positive and purity in (0, 1]")

    matrices = []
    activity_labels = []
    subject_ids = []
    window_rows = []
    source_hashes = {}
    feature_names = None
    candidate_count = 0
    for subject in SUBJECTS:
        path = source_dir / f"mHealth_subject{subject}.log"
        raw = load_subject(path)
        starts, majority, purity, selected = label_windows(raw[:, -1].astype(int),
                                                             window_size, stride, min_purity)
        candidate_count += len(starts)
        if not selected.any():
            raise ValueError(f"{path}: no windows meet the label policy")
        signals = prepare_signals(raw, mode)
        selected_starts = starts[selected]
        features, names = extract_features(signals, selected_starts, window_size)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("feature order changed between subjects")
        matrices.append(features)
        activity_labels.append(majority[selected])
        subject_ids.extend([subject] * len(features))
        first_data_row = len(subject_ids) - len(features)
        selected_index = 0
        for start, label, score, keep in zip(starts, majority, purity, selected):
            data_row = first_data_row + selected_index if keep else -1
            selected_index += int(keep)
            window_rows.append((subject, int(start), int(start + window_size),
                                int(label), float(score), int(keep), data_row))
        source_hashes[path.name] = _sha256(path)

    data = np.vstack(matrices)
    labels = np.concatenate(activity_labels).astype(int)
    subjects = np.asarray(subject_ids, dtype=int)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "data.txt", data, fmt="%.12g")
    np.savetxt(output_dir / "labels.txt", labels, fmt="%d")
    np.savetxt(output_dir / "subject_ids.txt", subjects, fmt="%d")
    np.savetxt(output_dir / "index_features.txt", np.arange(len(feature_names)), fmt="%d")
    old_target_index = output_dir / "index_target.txt"
    if old_target_index.exists():
        old_target_index.unlink()
    (output_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n")
    with (output_dir / "windows.csv").open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("subject_id", "start_sample", "stop_sample_exclusive",
                         "majority_label", "purity", "selected", "data_row"))
        writer.writerows(window_rows)
    for fold, subject in enumerate(SUBJECTS):
        np.savetxt(output_dir / f"index_train_{fold}.txt",
                   np.flatnonzero(subjects != subject), fmt="%d")
        np.savetxt(output_dir / f"index_test_{fold}.txt",
                   np.flatnonzero(subjects == subject), fmt="%d")
    np.savetxt(output_dir / "n_splits.txt", [len(SUBJECTS)], fmt="%d")
    metadata = {
        "schema_version": 2, "numpy_version": np.__version__,
        "features_file": "data.txt", "label_file": "labels.txt",
        "mode": mode, "sample_rate_hz": SAMPLE_RATE, "window_size": window_size,
        "stride": stride, "min_label_purity": min_purity,
        "selection": "majority activity 1-12 with purity >= threshold",
        "denoise_filter": {"kind": "zero-phase Hamming FIR", "cutoff_hz": 20.0, "taps": 101},
        "gravity_filter": {"kind": "zero-phase Hamming FIR", "cutoff_hz": 0.3, "taps": 501},
        "acceleration_unit": "g", "gyroscope_unit": "rad/s",
        "frequency_bands_hz": FREQUENCY_BANDS,
        "n_candidate_windows": candidate_count, "n_selected_windows": len(data),
        "n_features": len(feature_names), "fold_test_subjects": list(SUBJECTS),
        "selected_by_subject": {str(subject): int(np.count_nonzero(subjects == subject))
                                for subject in SUBJECTS},
        "selected_by_activity": {str(label): int(np.count_nonzero(labels == label))
                                 for label in range(1, 13)},
        "source_sha256": source_hashes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{mode}: {len(data)} / {candidate_count} windows selected; {len(feature_names)} features")
    print(f"output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "original")
    parser.add_argument("--mode", choices=MODE_NAMES, default="full")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--min-purity", type=float, default=1.0)
    args = parser.parse_args()
    dataset_name = "mhealth" if args.mode == "full" else "mhealth_acc_gyro_only"
    output_dir = args.output_dir or DATA_ROOT / dataset_name / "data"
    prepare(args.source_dir, output_dir, args.mode, WINDOW_SIZE,
            args.stride, args.min_purity)


if __name__ == "__main__":
    main()
