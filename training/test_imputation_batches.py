"""Numerical, memory-bound, checkpoint and training regressions for M3 batching."""

import copy
from pathlib import Path
import pickle
import tempfile
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.gnn_model import GNNStack
from models.egcn import EGCNConv
from models.prediction_model import ImputeNet
from training.imputation_batches import train_imputation_batches, predict_imputation_batches
from training.gnn_mdi import train_gnn_mdi
from uci.uci_data import get_data
from uci.raw_data.har.generate_m3_representation import load_module, impute_missing


def setUpModule():
    torch.set_num_threads(1)


def fixed_peers(self, mask, count, rows):
    # Keep peers identical across chunk boundaries for numerical comparison;
    # stochastic sampling/dropout legitimately changes the random sequence.
    return (rows[:, None] + torch.arange(self.sample_peer_size)[None, :] + 1) % self.num_record


def fixture(attr=True, peer=True, features=7, targets=13, hidden=4):
    records = 6
    model = ImputeNet(hidden_dim=hidden, num_of_record=records,
                      num_of_feature=features, device=torch.device('cpu'),
                      num_sample_peer=2, apply_relation=attr, apply_peer=peer,
                      sample_strategy='random_sample', drop_p=0)
    model.SRU.sample_peer = MethodType(fixed_peers, model.SRU)
    # SRU's two context units have their own dropout defaults.
    model.eval()
    rows = torch.arange(targets) % records
    cols = torch.arange(targets) % features
    edges = torch.stack((torch.cat((rows, records + cols)),
                         torch.cat((records + cols, rows))))
    mask = (torch.rand(records, features) > .3).float()
    labels = torch.rand(targets)
    known = torch.arange(targets) % 3 == 0
    return model, edges, mask, labels, known, records


class DecoderBatchTests(unittest.TestCase):
    def test_predictions_losses_and_all_gradients_match_legacy(self):
        # Exercise all context branches, both losses, and an uneven last batch.
        for attr, peer in ((False, False), (True, False), (False, True), (True, True)):
            for loss_mode in (0, 1):
                with self.subTest(attr=attr, peer=peer, loss_mode=loss_mode):
                    torch.manual_seed(42)
                    imputer, edges, mask, labels, known, records = fixture(attr, peer)
                    batched = copy.deepcopy(imputer)
                    encoder = nn.Linear(5, 4)
                    other_encoder = copy.deepcopy(encoder)
                    inputs = torch.rand(records + mask.shape[1], 5)
                    embedding = encoder(inputs)
                    pred, _ = imputer(embedding[:records], embedding[records:], edges, edges, mask)
                    expected = pred[:len(labels), 0]
                    if loss_mode:
                        expected = torch.where(known, labels, expected)
                    loss = (expected - labels).square().mean()
                    loss.backward()
                    actual, actual_loss = train_imputation_batches(
                        other_encoder(inputs), batched, edges, edges, mask, labels,
                        records, 5, known, loss_mode)
                    torch.testing.assert_close(actual, expected.detach())
                    self.assertAlmostEqual(actual_loss, loss.item(), places=6)
                    for a, b in zip(encoder.parameters(), other_encoder.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-7)
                    for (name, a), (_, b) in zip(imputer.named_parameters(), batched.named_parameters()):
                        if a.grad is None:
                            self.assertIsNone(b.grad, name)
                        else:
                            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-7,
                                                       msg=lambda m: name + ': ' + m)

    def test_decoder_activations_are_bounded_and_encoder_backward_runs_once(self):
        torch.manual_seed(42)
        imputer, edges, mask, labels, known, records = fixture(features=561, targets=37, hidden=12)
        embedding = torch.rand(records + 561, 12, requires_grad=True)
        seen, backward = [], []
        embedding.register_hook(lambda gradient: backward.append(gradient.clone()))
        hook = imputer.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[3].shape[1]))
        saved = []
        def pack(tensor):
            saved.append(tuple(tensor.shape))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            result, _ = train_imputation_batches(embedding, imputer, edges, edges, mask,
                                                 labels, records, 8)
        hook.remove()
        self.assertEqual(seen, [8, 8, 8, 8, 5])
        self.assertEqual(len(backward), 1)
        self.assertEqual(result.device.type, 'cpu')
        # Decoder inputs are bounded by B, including the SRU B*K*2F input.
        self.assertNotIn((37, 561), saved)
        self.assertNotIn((37, 2, 561), saved)
        self.assertIn((16, 1122), saved)

    def test_evaluation_and_inference_predict_unique_targets_in_order(self):
        torch.manual_seed(42)
        imputer, edges, mask, labels, _, records = fixture()
        embedding = torch.rand(records + 7, 4)
        expected, _ = imputer(embedding[:records], embedding[records:], edges, edges, mask)
        seen = []
        hook = imputer.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[3].shape[1]))
        actual = predict_imputation_batches(embedding, imputer, edges, edges, mask, records, 5)
        hook.remove()
        torch.testing.assert_close(actual, expected[:len(labels)].detach())
        self.assertEqual(seen, [5, 5, 3])
        self.assertFalse(actual.requires_grad)
        with self.assertRaises(ValueError):
            predict_imputation_batches(embedding, imputer, edges, edges, mask, records, 0)

    def test_cosine_cache_is_detached_and_valid_after_negative_vectors(self):
        for large in (False, True):
            features = torch.randn(7, 4, requires_grad=True)
            dense = torch.randn(6, 7, requires_grad=True)
            imputer = ImputeNet(hidden_dim=4, num_of_record=6, num_of_feature=7,
                                device=torch.device('cpu'), num_sample_peer=2,
                                apply_peer=True, sample_strategy='cos-similarity',
                                running_on_very_large_dataset=large, sample_space_size=4,
                                cos_sample_feature_embs=features, attn_score=dense)
            probabilities = imputer.SRU.sample_prob
            self.assertFalse(probabilities.requires_grad)
            self.assertIsNone(probabilities.grad_fn)
            self.assertTrue(bool((probabilities >= 0).all()))
            self.assertTrue(bool((probabilities.sum(1) > 0).all()))
            imputer.SRU.sample_peer(None, 6, torch.arange(6))


def small_gnn():
    return GNNStack(7, 1, 4, 3, 1, ['EGSAGE'] * 3, 0, 'relu', False,
                    [4], [True] * 3, 'mean', EPSILON=1e-4,
                    device=torch.device('cpu'), number_of_obs_nodes=6)


class EncoderAndTrainingTests(unittest.TestCase):
    def test_native_degree_reduction_preserves_weighted_gcn_normalization(self):
        edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        weights = torch.tensor([1., 2., 1., 3., 2., 3.], requires_grad=True)
        _, norm = EGCNConv.norm(edges, 4, weights)
        # Node 3 is isolated; its zero degree must not introduce NaNs.
        degrees = torch.stack((weights[0] + weights[1],
                               weights[2] + weights[3],
                               weights[4] + weights[5]))
        expected = weights / (degrees[edges[0]] * degrees[edges[1]]).sqrt()
        torch.testing.assert_close(norm, expected)
        actual_grad, = torch.autograd.grad(norm.sum(), weights, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected.sum(), weights)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_feature_parameter_updates_and_legacy_checkpoint_preserves_values(self):
        model = small_gnn()
        self.assertIn('feature_nodes', dict(model.named_parameters()))
        self.assertIn('feature_nodes', model.state_dict())
        original = model.feature_nodes.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        optimizer.zero_grad()
        model.node_init(torch.ones(6, 7)).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(original, model.feature_nodes.detach()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            torch.save(model, path)
            loaded = load_module(path, torch.device('cpu'))
            self.assertIsInstance(loaded.feature_nodes, nn.Parameter)
            torch.testing.assert_close(loaded.feature_nodes, model.feature_nodes)
            # Recreate the old complete-module layout, then load through the
            # real pickle hook and inference loader without altering values.
            values = model.feature_nodes.detach().clone()
            del model.feature_nodes
            model.feature_nodes = values.requires_grad_(True)
            torch.save(model, path)
            legacy = load_module(path, torch.device('cpu'))
            self.assertIsInstance(legacy.feature_nodes, nn.Parameter)
            torch.testing.assert_close(legacy.feature_nodes, values)
            torch.testing.assert_close(legacy.node_init(torch.ones(6, 7)),
                                       model.node_init(torch.ones(6, 7)))

    def test_two_epoch_real_gnn_training_validation_checkpoint_and_imputation(self):
        self._training_round_trip(features=7, valid=.15, strategy='cos-similarity',
                                  loss_mode=1, batch_size=5, peers=2)

    def test_har_width_random_peers_without_validation(self):
        # Match the user's 12/10 + FCU/SRU path with all 561 input features.
        self._training_round_trip(features=561, valid=0., strategy='random_sample',
                                  loss_mode=0, batch_size=128, peers=5)

    def _training_round_trip(self, features, valid, strategy, loss_mode, batch_size, peers):
        torch.manual_seed(42)
        args = SimpleNamespace(
            model_types='EGSAGE_EGSAGE_EGSAGE', norm_embs=None, post_hiddens=None,
            node_dim=12, edge_dim=10, edge_mode=1, dropout=0., gnn_activation='relu',
            concat_states=False, aggr='mean', init_epsilon=1e-4, impute_hiddens='12',
            valid=valid, auto_known=False, sample_peer_size=peers, apply_peer=True,
            apply_attr=True, sample_strategy=strategy, impute_nn_dropout=0.1,
            very_large_dataset=True, sample_space_size=4, opt='adam', weight_decay=0.,
            lr=.001, opt_scheduler='none', epochs=2, update_cos_sample_prob_every=1,
            known=.7, masking_distribution='uniform', loss_mode=loss_mode, impute_batch_size=batch_size,
            corrupt='mcar', save_model=True, save_prediction=True, mode='train', seed=42)
        x = np.random.default_rng(42).random((12, features))
        data = get_data(pd.DataFrame(x), pd.Series(np.arange(12) % 2), 0, .7, 0.,
                        'y', .7, seed=42, args=args)
        with tempfile.TemporaryDirectory() as folder:
            train_gnn_mdi(data, args, folder + '/', 0)
            with open(Path(folder) / 'result.pkl', 'rb') as handle:
                result = pickle.load(handle)
            self.assertEqual(len(result['curves']['train_loss']), 2)
            self.assertTrue(np.isfinite(result['curves']['train_loss']).all())
            self.assertEqual(result['outputs']['final_pred_train'].shape,
                             result['outputs']['label_train'].shape)
            model = load_module(Path(folder) / 'model.pt', torch.device('cpu'))
            imputer = load_module(Path(folder) / 'impute_model.pt', torch.device('cpu'))
            normalized = data.df_X.to_numpy(dtype=np.float32)
            missing = np.zeros_like(normalized, dtype=bool)
            missing[0, 0] = missing[4, 2] = missing[8, 6] = True
            dropped = normalized.copy()
            dropped[missing] = np.nan
            completed = impute_missing(model, imputer, dropped, missing,
                                       torch.device('cpu'), 1e-4, 2)
            self.assertTrue(np.isfinite(completed).all())
            np.testing.assert_array_equal(completed[~missing], normalized[~missing])


if __name__ == '__main__':
    unittest.main()
