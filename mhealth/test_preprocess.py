import tempfile
import unittest
from pathlib import Path

import numpy as np

from mhealth.preprocess import (WINDOW_SIZE, _spectral_features, extract_features, filter_continuous,
                                fir_lowpass, label_windows, prepare,
                                prepare_signals)


class PreprocessTests(unittest.TestCase):
    def test_fixed_windows_ignore_activity_boundaries(self):
        labels = np.r_[np.ones(64, dtype=int), np.full(64, 2), np.zeros(64, dtype=int)]
        starts, majority, purity, selected = label_windows(labels, 128, 64, 1.0)
        np.testing.assert_array_equal(starts, [0, 64])
        np.testing.assert_array_equal(majority, [1, 0])
        np.testing.assert_allclose(purity, [0.5, 0.5])
        self.assertFalse(selected.any())

    def test_filter_preserves_dc_and_removes_high_frequency_motion(self):
        t = np.arange(2048) / 50
        values = np.column_stack((np.ones(len(t)), np.sin(2 * np.pi * 5 * t)))
        filtered = filter_continuous(values, fir_lowpass(0.3, 501))
        np.testing.assert_allclose(filtered[600:-600, 0], 1, atol=1e-10)
        self.assertLess(np.sqrt(np.mean(filtered[600:-600, 1] ** 2)), 0.01)

    def test_frequency_band_and_unit_conversion(self):
        t = np.arange(WINDOW_SIZE) / 50
        spectral = _spectral_features(np.sin(2 * np.pi * 5 * t)[None, :])
        self.assertAlmostEqual(spectral["dominant_hz"][0], 5, delta=0.4)
        self.assertGreater(spectral["power_3_8_fraction"][0], 0.95)

        raw = np.zeros((WINDOW_SIZE * 2, 24))
        raw[:, 0:3] = 9.80665
        raw[:, 8:11] = 180
        signals = prepare_signals(raw, "acc_gyro_only")
        np.testing.assert_allclose(signals["gravity_acc_chest"], 1, atol=1e-9)
        np.testing.assert_allclose(signals["body_acc_chest"], 0, atol=1e-9)
        np.testing.assert_allclose(signals["gyro_ankle"], np.pi, atol=1e-9)

    def test_modes_produce_finite_named_features(self):
        t = np.arange(WINDOW_SIZE * 2) / 50
        raw = np.zeros((len(t), 24))
        raw[:, :23] = np.sin(2 * np.pi * 2 * t)[:, None]
        for mode, expected in (("acc_gyro_only", 253), ("full", 295)):
            signals = prepare_signals(raw, mode)
            features, names = extract_features(signals, np.array([0, 64, 128]), WINDOW_SIZE)
            self.assertEqual(features.shape, (3, expected))
            self.assertEqual(len(names), expected)
            self.assertEqual(len(set(names)), expected)
            self.assertTrue(np.isfinite(features).all())
            self.assertEqual("ecg_1_mean" in names, mode == "full")

    def test_generated_splits_are_subject_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            output = Path(temp) / "output"
            source.mkdir()
            for subject in range(1, 11):
                raw = np.zeros((WINDOW_SIZE * 2, 24))
                raw[:, :23] = subject
                raw[:, -1] = 1
                np.savetxt(source / f"mHealth_subject{subject}.log", raw)
            prepare(source, output, "acc_gyro_only", WINDOW_SIZE, 64, 1.0)
            data = np.loadtxt(output / "data.txt")
            labels = np.loadtxt(output / "labels.txt", dtype=int)
            subjects = np.loadtxt(output / "subject_ids.txt", dtype=int)
            self.assertEqual(data.shape, (30, 253))
            np.testing.assert_array_equal(labels, np.ones(30, dtype=int))
            self.assertFalse((output / "index_target.txt").exists())
            for fold, subject in enumerate(range(1, 11)):
                train = np.loadtxt(output / f"index_train_{fold}.txt", dtype=int)
                test = np.loadtxt(output / f"index_test_{fold}.txt", dtype=int)
                self.assertTrue(np.all(subjects[test] == subject))
                self.assertTrue(np.all(subjects[train] != subject))


if __name__ == "__main__":
    unittest.main()
