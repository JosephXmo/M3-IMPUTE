"""Bound decoder memory while keeping one full-graph optimizer step per epoch."""

import torch
import torch.nn.functional as F


def _check_batch_size(batch_size):
    if batch_size < 1:
        raise ValueError("impute_batch_size must be positive")


def train_imputation_batches(embeddings, imputer, known_edges, target_edges,
                             known_mask, labels, num_records, batch_size,
                             known_targets=None, loss_mode=0, ce_loss=False):
    """Backpropagate each decoder batch, then the encoder exactly once.

    A detached leaf collects dL/d(encoder output). Decoder intermediates are
    released after each batch backward. Finally the accumulated leaf gradient
    is passed to the original encoder graph. Merely concatenating differentiable
    batch predictions and calling backward at the end would retain every batch.
    Sum losses are divided by the total target count, including the last short
    batch; loss_mode=1 keeps the legacy all-target denominator.
    """
    _check_batch_size(batch_size)
    count = len(labels)
    if count == 0 or target_edges.shape[1] != 2 * count:
        raise ValueError("training requires nonempty paired edges aligned with labels")
    if loss_mode not in (0, 1):
        raise ValueError("loss_mode must be 0 or 1")
    if loss_mode == 1 and known_targets is None:
        raise ValueError("hidden-only loss requires the known-target mask")
    proxy = embeddings.detach().requires_grad_(True)
    predictions = None
    total_loss = 0.0
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        # Only sample->feature targets enter the decoder. Bidirectional context
        # edges still enter the encoder; reverse predictions never entered loss.
        pred, _ = imputer(proxy[:num_records], proxy[num_records:], known_edges,
                          target_edges[:, start:stop], known_mask)
        target = labels[start:stop]
        if ce_loss:
            per_target = F.cross_entropy(pred, target, reduction="none")
            if loss_mode == 1:
                per_target = per_target * (~known_targets[start:stop])
            loss = per_target.sum() / count
            reported = pred
        else:
            reported = pred[:, 0]
            if loss_mode == 1:
                reported = torch.where(known_targets[start:stop], target, reported)
            loss = F.mse_loss(reported, target, reduction="sum") / count
        if predictions is None:
            predictions = torch.empty((count, *reported.shape[1:]), dtype=reported.dtype, device='cpu')
        predictions[start:stop] = reported.detach().cpu()
        loss.backward()
        total_loss += loss.detach().item()
        del pred, reported, loss
    embeddings.backward(proxy.grad)
    return predictions, total_loss


@torch.no_grad()
def predict_imputation_batches(embeddings, imputer, known_edges, target_edges,
                               known_mask, num_records, batch_size):
    """Evaluate each unique target once; keep full predictions only on CPU."""
    _check_batch_size(batch_size)
    if target_edges.shape[1] % 2:
        raise ValueError("evaluation requires paired directed target edges")
    count = target_edges.shape[1] // 2
    predictions = None
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        pred, _ = imputer(embeddings[:num_records], embeddings[num_records:], known_edges,
                          target_edges[:, start:stop], known_mask)
        if predictions is None:
            predictions = torch.empty((count, pred.shape[1]), dtype=pred.dtype, device='cpu')
        predictions[start:stop] = pred.cpu()
    return predictions if predictions is not None else torch.empty((0, 1), device='cpu')
