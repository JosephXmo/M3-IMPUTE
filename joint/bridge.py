"""Exact decoder recomputation/VJP bridge with bounded decoder activations.

No-grad predictions alone are not joint training: replay_backward explicitly
passes dL/d(reconstruction) through each decoder chunk, then the live encoder.
Stochastic CPU/CUDA RNG states are replayed so dropout/peer samples match.
The encoder and downstream graphs remain full graphs, with one optimizer step.
"""

from dataclasses import dataclass

import torch


def target_edges(start, stop, rows, features, device):
    indices = torch.arange(start, stop, device=device)
    return torch.stack((indices // features, rows + indices % features))


@dataclass
class DecoderTape:
    values: torch.Tensor
    states: list
    batch_size: int
    training: bool


def cuda_devices(device):
    return [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []


@torch.no_grad()
def decode_all(embeddings, imputer, known_edges, known_mask, batch_size, record_rng=False):
    """Decode ALL cells in row-major order, including currently observed cells."""
    if batch_size < 1:
        raise ValueError('decoder batch_size must be positive')
    n, f = known_mask.shape
    output = embeddings.new_empty(n * f)
    states = []
    for start in range(0, n * f, batch_size):
        stop = min(start + batch_size, n * f)
        if record_rng:
            states.append((torch.get_rng_state(), [torch.cuda.get_rng_state(d) for d in cuda_devices(embeddings.device)]))
        targets = target_edges(start, stop, n, f, embeddings.device)
        predicted, _ = imputer(embeddings[:n], embeddings[n:], known_edges, targets, known_mask)
        if predicted.shape != (stop - start, 1):
            raise ValueError('Expected M3 scalar regression output')
        output[start:stop] = predicted[:, 0]
    return DecoderTape(output.reshape(n, f), states, batch_size, imputer.training)


def replay_backward(embeddings, imputer, known_edges, known_mask, tape, output_gradient):
    """Accumulate decoder parameter gradients and propagate one encoder VJP.

    Caller already ran downstream backward on a detached leaf containing
    tape.values. No parameters, masks, caches, or train/eval modes may change
    between decode_all and replay_backward. No optimizer step occurs here.
    """
    n, f = known_mask.shape
    expected = (n * f + tape.batch_size - 1) // tape.batch_size
    if len(tape.states) != expected or output_gradient.shape != (n, f):
        raise ValueError('Replay requires matching gradient and recorded RNG tape')
    if imputer.training != tape.training:
        raise ValueError('Decoder mode changed between forward and replay')
    proxy = embeddings.detach().requires_grad_(True)
    gradient = output_gradient.detach().reshape(-1)
    devices = cuda_devices(embeddings.device)
    for chunk, start in enumerate(range(0, n * f, tape.batch_size)):
        stop = min(start + tape.batch_size, n * f)
        cpu_state, gpu_states = tape.states[chunk]
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_state)
            for device, state in zip(devices, gpu_states):
                torch.cuda.set_rng_state(state, device)
            targets = target_edges(start, stop, n, f, embeddings.device)
            predicted, _ = imputer(proxy[:n], proxy[n:], known_edges, targets, known_mask)
            predicted[:, 0].backward(gradient[start:stop])
    if proxy.grad is None:
        raise RuntimeError('Decoder did not produce an embedding gradient')
    embeddings.backward(proxy.grad)
