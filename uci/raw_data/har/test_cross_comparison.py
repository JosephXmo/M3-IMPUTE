"""Focused protocol checks; no HAR training or GPU is required."""

import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import preprocess_cross_comparison as pipeline


class ProtocolTests(unittest.TestCase):
    def test_crossing_subject_moves_entirely_to_s0(self):
        subjects = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
        train, info = pipeline.subject_split(subjects, 0.7)
        self.assertEqual(info["moved_subjects"], [])
        train, info = pipeline.subject_split(subjects, 0.6)
        self.assertEqual(info["moved_subjects"], [3])
        self.assertEqual(info["train_rows"], 4)
        self.assertFalse(set(subjects[train]) & set(subjects[~train]))
        self.assertGreaterEqual(info["s0_rows"], 4)

    def test_statistics_match_sklearn_including_constant_columns(self):
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("sklearn not available")
        from sklearn.preprocessing import MinMaxScaler
        x = np.array([[1., 3., 0.], [2., 3., 1e-17], [4., 3., 2e-17]])
        stats = pipeline.training_statistics(x)
        scaler = MinMaxScaler().fit(x)
        np.testing.assert_allclose(stats["minmax_scale"], scaler.scale_)
        np.testing.assert_allclose(stats["minmax_offset"], scaler.min_)

    def test_prepare_uses_train_only_statistics_and_deterministic_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, training = root / "source", root / "output", root / "training"
            x = np.repeat(np.arange(10, dtype=float)[:, None], 561, axis=1)
            subjects = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
            for name, sl in (("train", slice(0, 7)), ("test", slice(7, 10))):
                pipeline.save_txt(source / name / f"X_{name}.txt", x[sl])
                pipeline.save_txt(source / name / f"y_{name}.txt", np.ones(len(x[sl])), True)
                pipeline.save_txt(source / name / f"subject_{name}.txt", subjects[sl], True)
            args = argparse.Namespace(output_dir=output, source=source, train_ratio=.6,
                                      rate=.2, seed=42, radius=5, min_overlap=8,
                                      side_margin=.1, side_ratio=1.5, update_xhar_configs=False)
            with patch.object(pipeline, "TRAIN_DATA", training):
                pipeline.prepare(args)
                first_mask = (output / "missing_mask.txt").read_bytes()
                pipeline.prepare(args)
            self.assertEqual(first_mask, (output / "missing_mask.txt").read_bytes())
            stats = pipeline.read_json(output / "train_statistics.json")
            np.testing.assert_allclose(stats["mean"], 1.5)
            s0 = np.loadtxt(output / "hhar_s0.txt")
            drop = np.loadtxt(output / "hhar_drop.txt")
            lerp = np.loadtxt(output / "hhar_lerp.txt")
            mask = np.loadtxt(output / "missing_mask.txt", dtype=bool)
            np.testing.assert_array_equal(drop[~mask], s0[~mask])
            np.testing.assert_array_equal(lerp[~mask], s0[~mask])
            np.testing.assert_allclose(drop[mask], 1.5)
            self.assertEqual(s0.shape, (6, 561))
            self.assertEqual(np.loadtxt(training / "data.txt").shape, (4, 562))
            # Remove all true/source data. Interpolation still runs identically.
            (output / "hhar_s0.txt").unlink()
            for path in source.rglob("*.txt"):
                path.unlink()
            pipeline.verify_artifacts(output, pipeline.read_json(output / "manifest.json"))
            pipeline.interpolate_artifacts(output, args)
            np.testing.assert_array_equal(np.loadtxt(output / "hhar_lerp.txt"), lerp)

    def interpolation_case(self, visible_feature, segments=None):
        x = np.column_stack((np.arange(5, dtype=float), visible_feature))
        mask = np.zeros_like(x, dtype=bool)
        mask[2, 0] = True
        x[2, 0] = 9999
        stats = pipeline.training_statistics(np.array([[0., -1.], [4., 1.]]))
        stats["std"] = [1., 1.]
        return x, mask, np.zeros(5, dtype=int) if segments is None else segments, stats

    def test_left_right_and_both_direction_selection(self):
        for feature, mode in (([0, 0, 0, 10, 10], "left"),
                              ([10, 10, 0, 0, 0], "right"),
                              ([0, 0, 0, 0, 0], "both")):
            args = self.interpolation_case(feature)
            result, counts = pipeline.similarity_lerp(*args, min_overlap=1)
            self.assertEqual(counts[mode], 1)
            self.assertAlmostEqual(result[2, 0], 2)
            np.testing.assert_array_equal(result[~args[1]], args[0][~args[1]])

    def test_masked_placeholders_do_not_affect_prediction(self):
        x, mask, segments, stats = self.interpolation_case([0] * 5)
        first, _ = pipeline.similarity_lerp(x, mask, segments, stats, min_overlap=1)
        x[mask] = np.nan
        second, _ = pipeline.similarity_lerp(x, mask, segments, stats, min_overlap=1)
        np.testing.assert_array_equal(first, second)

    def test_no_cross_segment_interpolation(self):
        args = self.interpolation_case([0] * 5, np.array([0, 0, 1, 2, 2]))
        result, counts = pipeline.similarity_lerp(*args, min_overlap=1)
        self.assertEqual(counts["mean"], 1)
        self.assertEqual(result[2, 0], args[3]["mean"][0])

    def test_config_patch_preserves_training_settings_and_comments(self):
        original = '# X_path: old\nX_path: "old"\ny_path: "old-label"\nseed: 42\ngraph_path: ""\n'
        patched = pipeline.config_text(original, "m3")
        a, b = original.splitlines(), patched.splitlines()
        self.assertEqual([i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]], [1, 2])
        self.assertIn('data/missing-cross-comparison/hhar_m3.txt', patched)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not available")
class TorchAdapterTests(unittest.TestCase):
    def test_actual_imputenet_on_new_row_count_all_context_modes(self):
        import torch
        from models.prediction_model import ImputeNet
        from generate_m3_representation import impute_missing

        class Encoder(torch.nn.Module):
            # Only the encoder is a toy here; indexing and prediction use the
            # real ImputeNet and graph/chunk helper. This is not GNN validation.
            def __init__(self):
                super().__init__()
                self.device = torch.device("cpu")
                self.number_of_feature = 3
                self.number_of_obs_node = 9
                self.feature_nodes = torch.ones(3, 4)

            def forward(self, nodes, edge_attr, edge_index, dense):
                return torch.cat((dense @ self.feature_nodes, self.feature_nodes))

        x = np.array([[.1, .2, np.nan], [.3, np.nan, .6], [np.nan, .8, .9], [.4, .5, .6]],
                     dtype=np.float32)
        missing = np.isnan(x)
        device = torch.device("cpu")
        for attr, peer in ((False, False), (True, False), (False, True), (True, True)):
            for strategy in ("random_sample", "cos-similarity"):
                model = Encoder()
                imputer = ImputeNet(hidden_dim=4, num_of_record=9, num_of_feature=3,
                                    device=device, num_sample_peer=2, apply_relation=attr,
                                    apply_peer=peer, sample_strategy=strategy,
                                    cos_sample_feature_embs=model.feature_nodes,
                                    attn_score=torch.ones(9, 3), drop_p=0.)
                imputer.eval()
                with torch.no_grad():
                    pipeline.adapt_checkpoint(model, imputer, 4, 3,
                                              torch.from_numpy(np.nan_to_num(x)), device)
                    result = impute_missing(model, imputer, x, missing, device, 1e-4, 1)
                self.assertEqual(imputer.SRU.num_record, 4)
                self.assertEqual(model.number_of_obs_node, 4)
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_array_equal(result[~missing], x[~missing])


if __name__ == "__main__":
    unittest.main()
