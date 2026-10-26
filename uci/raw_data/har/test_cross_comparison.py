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
    def test_random_split_is_disjoint_complete_reproducible_and_shuffled(self):
        subjects = np.tile(np.arange(1, 31), 344)[:10299]
        train, s0, info = pipeline.random_row_split(subjects, .7, 42)
        self.assertEqual((len(train), len(s0)), (7209, 3090))
        np.testing.assert_array_equal(np.sort(np.r_[train, s0]), np.arange(10299))
        self.assertFalse(np.intersect1d(train, s0).size)
        self.assertFalse(np.all(np.diff(train) > 0))
        self.assertFalse(np.all(np.diff(s0) > 0))
        again = pipeline.random_row_split(subjects, .7, 42)
        np.testing.assert_array_equal(train, again[0])
        np.testing.assert_array_equal(s0, again[1])
        self.assertFalse(np.array_equal(train, pipeline.random_row_split(subjects, .7, 43)[0]))
        self.assertEqual(info["shared_subjects"], list(range(1, 31)))

    def test_invalid_or_empty_split_is_rejected(self):
        for ratio in (0, 1, -.1, 1.1, .01):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                pipeline.random_row_split(np.arange(10), ratio, 42)

    def test_statistics_match_sklearn_including_constant_columns(self):
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("sklearn not available")
        from sklearn.preprocessing import MinMaxScaler
        x = np.array([[1., 3., 0.], [2., 3., 1e-17], [4., 3., 2e-17]])
        stats = pipeline.training_statistics(x)
        scaler = MinMaxScaler().fit(x)
        np.testing.assert_allclose(stats["minmax_scale"], scaler.scale_)
        np.testing.assert_allclose(stats["minmax_offset"], scaler.min_)

    def test_prepare_alignment_statistics_masks_and_retired_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, training = root / "source", root / "output", root / "training"
            x = np.repeat(np.arange(10, dtype=float)[:, None], 561, axis=1)
            labels = np.arange(10) % 6 + 1
            subjects = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
            for name, sl in (("train", slice(0, 7)), ("test", slice(7, 10))):
                pipeline.save_txt(source / name / f"X_{name}.txt", x[sl])
                pipeline.save_txt(source / name / f"y_{name}.txt", labels[sl], True)
                pipeline.save_txt(source / name / f"subject_{name}.txt", subjects[sl], True)
            args = argparse.Namespace(output_dir=output, source=source, train_ratio=.7,
                                      rate=.2, seed=42, update_xhar_configs=False)
            for name in ("hhar_lerp.txt", "segments.txt", "hhar_m3.txt"):
                pipeline.save_txt(output / name, np.array([123.]))
            with patch.object(pipeline, "TRAIN_DATA", training):
                pipeline.prepare(args)
                first_mask = (output / "missing_mask.txt").read_bytes()
                first_manifest = pipeline.read_json(output / "manifest.json")
                pipeline.prepare(args)
            self.assertEqual(first_mask, (output / "missing_mask.txt").read_bytes())
            self.assertEqual(first_manifest, pipeline.read_json(output / "manifest.json"))
            train_ids = np.loadtxt(output / "train_row_ids.txt", dtype=int)
            s0_ids = np.loadtxt(output / "s0_row_ids.txt", dtype=int)
            np.testing.assert_array_equal(np.sort(np.r_[train_ids, s0_ids]), np.arange(10))
            np.testing.assert_array_equal(np.loadtxt(training / "data.txt"),
                                          np.column_stack((x[train_ids], labels[train_ids])))
            np.testing.assert_array_equal(np.loadtxt(output / "hhar_labels.txt"), labels[s0_ids])
            np.testing.assert_array_equal(np.loadtxt(output / "s0_subjects.txt"), subjects[s0_ids])
            stats = pipeline.read_json(output / "train_statistics.json")
            np.testing.assert_allclose(stats["mean"], x[train_ids].mean(0))
            s0 = np.loadtxt(output / "hhar_s0.txt")
            np.testing.assert_array_equal(s0, x[s0_ids])
            drop = np.loadtxt(output / "hhar_drop.txt")
            mask = np.loadtxt(output / "missing_mask.txt", dtype=bool)
            np.testing.assert_array_equal(drop[~mask], s0[~mask])
            np.testing.assert_allclose(drop[mask], x[train_ids].mean())
            missing = np.loadtxt(output / "hhar_missing.txt")
            np.testing.assert_array_equal(np.isnan(missing), mask)
            np.testing.assert_array_equal(missing[~mask], s0[~mask])
            self.assertEqual(s0.shape, (3, 561))
            self.assertEqual(set(first_manifest["conditions"]), {"s0", "drop", "m3"})
            for name in ("hhar_lerp.txt", "segments.txt", "hhar_m3.txt"):
                self.assertFalse((output / name).exists())
            self.assertTrue(list(output.glob("hhar_lerp.retired-*.txt")))
            self.assertTrue(list(output.glob("hhar_m3.stale-*.txt")))
            pipeline.verify_artifacts(output, first_manifest)
            # Rate changes leave the split and its training checkpoint binding intact.
            args.rate = 1.
            with patch.object(pipeline, "TRAIN_DATA", training):
                pipeline.prepare(args)
            changed = pipeline.read_json(output / "manifest.json")
            self.assertEqual(changed["train_sha256"], first_manifest["train_sha256"])
            self.assertEqual(changed["s0_sha256"], first_manifest["s0_sha256"])
            self.assertEqual(changed["actual_missing_rate"], 1.)
            # Row-order metadata must remain bound to the prepared artifacts.
            pipeline.save_txt(output / "s0_row_ids.txt", s0_ids[::-1], True)
            with self.assertRaisesRegex(ValueError, "s0_row_ids"):
                pipeline.verify_artifacts(output, changed)

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
