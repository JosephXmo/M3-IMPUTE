"""Full-population denoising M3 -> Juicer workflow, with explicit updates."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import resource
import time

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import torch_geometric

from xhar_juicer import sdcn_loss, target_distribution, rope_from_positions, temporal_loss
from .bridge import decode_all, replay_backward, cuda_devices
from .data import load_har, bipartite_graph, visibility_mask, make_xhar_graph, sha256
from .metrics import evaluate_clusters
from .model import JointModel


DEFAULTS = dict(
    seed=42, epochs=2, device='cpu', threads=4, mask_rate=.1,
    m3_hidden=4, m3_layers=1, m3_edge_dim=4, m3_dropout=0., fcu=False, sru=False, peers=2,
    decoder_batch_size=16384, xhar_widths=[32, 32, 64], xhar_edge_dim=8,
    latent=6, clusters=6, rope_dim=4, xhar_edge_mode=1, rope_warm_epochs=10,
    ae_epochs=1, ae_batch_size=256, kmeans_n_init=2, lr=.001,
    m3_weight=1., xhar_weight=1., temporal_weight=.1,
    topk=5, similarity_threshold=.988, graph_block_size=256, temporal_delta=2,
    graph_refresh_interval=1, eval_seed=20261027,
)


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError(f'Unknown or missing config keys: {set(config) ^ set(DEFAULTS)}')
    for key in ('epochs', 'threads', 'm3_hidden', 'm3_layers', 'm3_edge_dim', 'peers',
                'decoder_batch_size', 'xhar_edge_dim', 'latent', 'clusters', 'rope_dim',
                'ae_batch_size', 'kmeans_n_init', 'topk', 'graph_block_size'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if len(config['xhar_widths']) != 3 or any(v < 1 for v in config['xhar_widths']):
        raise ValueError('xhar_widths requires three positive sizes')
    if config['rope_dim'] % 2 or not 0 < config['mask_rate'] < 1:
        raise ValueError('rope_dim must be even; mask_rate must lie in (0,1)')
    if config['ae_epochs'] < 0 or config['graph_refresh_interval'] < 0 or config['temporal_delta'] < 0:
        raise ValueError('phase/graph intervals must be nonnegative')
    if config['m3_weight'] < 0 or config['xhar_weight'] <= 0 or config['lr'] <= 0:
        raise ValueError('Invalid loss weights or learning rate')
    if not 0 <= config['m3_dropout'] < 1 or config['xhar_edge_mode'] not in range(6):
        raise ValueError('Invalid dropout or edge mode')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def make_context(data, known_cpu, device):
    target = data.normalized.to(device)
    known = known_cpu.to(device)
    visible = torch.where(known, target, 1e-4)
    edges, attrs = bipartite_graph(visible, known)
    return target, known, visible, edges, attrs


def module_gradients(model):
    result = {}
    for name in ('encoder', 'imputer', 'juicer', 'edge_features'):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f'{name}: missing or non-finite parameter gradients')
        result[name] = sum(g.detach().double().square().sum().item() for g in grads) ** .5
        if result[name] == 0:
            raise RuntimeError(f'{name}: zero gradient; joint path is disconnected or degenerate')
    return result


def graph_from_reconstruction(values, data, config, device):
    # Topology is discrete; it sees ONLY M3 reconstructed features, never truth.
    raw = values.detach().cpu() * data.scale + data.offset
    edges, pairs = make_xhar_graph(
        raw, data.segments, config['topk'], config['similarity_threshold'],
        config['graph_block_size'], config['temporal_delta'],
    )
    edges, pairs = edges.to(device), pairs.to(device)
    rope = rope_from_positions(torch.arange(len(raw), device=device).float(), edges, config['rope_dim'])
    return edges, pairs, rope


@torch.no_grad()
def reconstruct(model, data, known, config, device):
    _, mask, visible, edges, attrs = make_context(data, known, device)
    embeddings = model.encode(visible, edges, attrs)
    return decode_all(embeddings, model.imputer, edges, mask.float(), config['decoder_batch_size']).values


def initialize(model, data, known, config, device):
    model.eval()
    log('Initial masked M3 reconstruction of all cells')
    with torch.random.fork_rng(devices=cuda_devices(device)):
        torch.manual_seed(config['eval_seed'])
        values = reconstruct(model, data, known, config, device)
    graph = graph_from_reconstruction(values, data, config, device)
    x = values * data.scale.to(device) + data.offset.to(device)
    optimizer = torch.optim.Adam(model.juicer.ae.parameters(), lr=config['lr'])
    model.juicer.ae.train()
    for epoch in range(config['ae_epochs']):
        total = 0.
        for batch in torch.randperm(len(x), device=device).split(config['ae_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model.juicer.ae(x[batch])[0], x[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        log(f'AE heatup {epoch + 1}/{config["ae_epochs"]}: MSE={total / len(x):.6f}')
    model.eval()
    with torch.no_grad():
        embedding = model.juicer.ae(x)[-1].cpu().numpy()
    kmeans = KMeans(n_clusters=config['clusters'], n_init=config['kmeans_n_init'], random_state=config['seed'])
    kmeans.fit(embedding)
    with torch.no_grad():
        model.juicer.cluster_layer.copy_(torch.as_tensor(kmeans.cluster_centers_, device=device))
    model.zero_grad(set_to_none=True)
    return graph


def train_epoch(model, optimizer, data, config, epoch, graph, device):
    started = time.monotonic()
    known_cpu = visibility_mask(data.normalized.shape, config['mask_rate'], config['seed'] + 1000 + epoch)
    target, known, visible, m3_edges, attrs = make_context(data, known_cpu, device)
    model.train()
    model.juicer.gat_1.rope_warmup(epoch - 1, config['rope_warm_epochs'])
    optimizer.zero_grad(set_to_none=True)
    log(f'Epoch {epoch}: M3 encode, {m3_edges.shape[1]:,} visible directed edges')
    embeddings = model.encode(visible, m3_edges, attrs)
    log(f'Epoch {epoch}: reconstruct {target.numel():,} cells')
    tape = decode_all(embeddings, model.imputer, m3_edges, known.float(), config['decoder_batch_size'], record_rng=True)
    if config['graph_refresh_interval'] and (epoch - 1) % config['graph_refresh_interval'] == 0:
        graph = graph_from_reconstruction(tape.values, data, config, device)
    edges, pairs, rope = graph
    proxy = tape.values.detach().requires_grad_(True)
    x, output = model.xhar(proxy, data.offset.to(device), data.scale.to(device), edges, rope)
    core = sdcn_loss(output, x.detach(), target_distribution(output.q))
    temporal = temporal_loss(output.prediction, pairs)
    downstream = core.total + config['temporal_weight'] * temporal
    # Measure the downstream gradient before adding M3 reconstruction loss.
    downstream_gradient, = torch.autograd.grad(downstream, proxy, retain_graph=True)
    downstream_norm = downstream_gradient.detach().double().norm().item()
    if not np.isfinite(downstream_norm) or downstream_norm == 0:
        raise RuntimeError('XHAR produced no finite nonzero gradient to M3 reconstruction')
    del downstream_gradient
    m3_loss = F.mse_loss(proxy[~known], target[~known])
    total = config['m3_weight'] * m3_loss + config['xhar_weight'] * downstream
    if not torch.isfinite(total):
        raise RuntimeError('Non-finite joint loss')
    log(f'Epoch {epoch}: XHAR backward and decoder/encoder gradient replay')
    total.backward()
    replay_backward(embeddings, model.imputer, m3_edges, known.float(), tape, proxy.grad)
    gradients = module_gradients(model)
    watched = {'encoder': model.encoder.feature_nodes, 'imputer': model.imputer.post_neural_net[-2].weight,
               'juicer': model.juicer.gat_1.agg_lin.weight}
    before = {name: p.detach().clone() for name, p in watched.items()}
    optimizer.step()
    updates = {name: (p.detach() - before[name]).double().norm().item() for name, p in watched.items()}
    if not all(value > 0 for value in updates.values()):
        raise RuntimeError('An expected encoder/decoder/Juicer parameter did not update')
    stats = {'epoch': epoch, 'loss': total.item(), 'm3_hidden_mse': m3_loss.item(),
             'xhar_core_loss': core.total.item(), 'temporal_loss': temporal.item(),
             'xhar_to_reconstruction_gradient_l2': downstream_norm, 'gradient_l2': gradients,
             'parameter_update_l2': updates, 'hidden_cells': int((~known).sum()),
             'm3_directed_edges': m3_edges.shape[1], 'xhar_edges': edges.shape[1],
             'seconds': time.monotonic() - started}
    return graph, stats


def evaluate(model, data, known, graph, config, epoch, device):
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices(device)):
        torch.manual_seed(config['eval_seed'])
        completed = reconstruct(model, data, known, config, device)
        x, output = model.xhar(completed, data.offset.to(device), data.scale.to(device), graph[0], graph[2])
        p = target_distribution(output.q)
        predictions = {name: values.argmax(1).cpu().numpy() for name, values in
                       [('Q', output.q), ('Z', output.prediction), ('P', p)]}
        target = data.normalized.to(device)
        hidden = ~known.to(device)
        rmse = F.mse_loss(completed[hidden], target[hidden]).sqrt().item()
    model.train(was_training)
    rows = [{'Epoch': epoch, 'Distribution': name, **evaluate_clusters(data.labels, values),
             'M3 hidden RMSE': rmse} for name, values in predictions.items()]
    main = next(row for row in rows if row['Distribution'] == 'Z')
    log(f'Epoch {epoch}: Z Accuracy={main["Accuracy"]:.4%}, NMI={main["NMI"]:.4f}, ARI={main["ARI"]:.4f}')
    return rows, predictions


def write_metrics(output, history, predictions, data):
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(output / 'predictions.npz', **predictions, labels=data.labels, subjects=data.subjects,
                        row_id=np.arange(len(data.features)))


def source_hashes():
    import xhar_juicer

    root = Path(__file__).resolve().parents[1]
    paths = [root / 'train_joint.py']
    for folder in ('joint', 'models', 'utils'):
        paths.extend(p for p in (root / folder).rglob('*.py')
                     if 'tests' not in p.parts and not p.name.startswith('test_')
                     and not p.name.endswith('_test.py'))
    hashes = {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}
    # Record the code actually imported, independent of where the manual copy
    # was installed. Local test files are not part of the training protocol.
    package = Path(xhar_juicer.__file__).resolve().parent
    hashes.update({f'xhar_juicer/{p.relative_to(package)}': sha256(p)
                   for p in sorted(package.rglob('*.py'))
                   if 'tests' not in p.relative_to(package).parts
                   and not p.name.startswith('test_') and not p.name.endswith('_test.py')})
    return hashes


def rng_state():
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def run(config, data_dir, output, resume=None, evaluate_only=False):
    validate_config(config)
    if evaluate_only and not resume:
        raise ValueError('--evaluate requires --resume checkpoint')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f'Choose an empty output directory: {output}')
    torch.set_num_threads(config['threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = torch.device(config['device'])
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('This runner supports CPU and CUDA; MPS replay is not implemented')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    log('Load complete official UCI-HAR train + test tables')
    data = load_har(data_dir)
    evaluation_mask = visibility_mask(data.normalized.shape, config['mask_rate'], config['eval_seed'])
    model = JointModel(len(data.features), data.features.shape[1], config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    manifest = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'config': config,
                'dataset': data.manifest, 'source_sha256': source_hashes(),
                'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                'pyg': torch_geometric.__version__, 'device': str(device)},
                'evaluation_hidden_cells': int((~evaluation_mask).sum()),
                'parameters': {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()},
                'resume': str(resume) if resume else None, 'evaluate_only': evaluate_only}
    write_json(output / 'run.json', manifest)
    write_json(output / 'config.json', config)
    np.savez_compressed(output / 'normalization.npz', offset=data.offset.numpy(), scale=data.scale.numpy())
    np.save(output / 'evaluation_known_mask.npy', evaluation_mask.numpy())
    started = time.monotonic()
    try:
        history, training, last_epoch = [], [], 0
        if resume:
            # Checkpoints contain optimizer and Python/NumPy RNG; load trusted own files only.
            checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
            if checkpoint['dataset'] != data.manifest:
                raise ValueError('Checkpoint dataset provenance mismatch')
            allowed = {'epochs', 'device', 'threads'}
            changes = [k for k in config if k not in allowed and config[k] != checkpoint['config'][k]]
            if changes:
                raise ValueError(f'Architecture/protocol changes require a new run: {changes}')
            if checkpoint['source_sha256'] != manifest['source_sha256']:
                raise ValueError('Source changed since checkpoint; retain the original code for an exact resume')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            graph = tuple(t.to(device) for t in checkpoint['graph'])
            history, training, last_epoch = checkpoint['metrics'], checkpoint['training'], checkpoint['epoch']
            restore_rng(checkpoint['rng'])
            del checkpoint
        else:
            graph = initialize(model, data, evaluation_mask, config, device)
        if evaluate_only:
            rows, predictions = evaluate(model, data, evaluation_mask, graph, config, last_epoch, device)
            write_metrics(output, rows, predictions, data)
        else:
            if last_epoch >= config['epochs']:
                raise ValueError('--epochs must exceed the checkpoint epoch when resuming training')
            for epoch in range(last_epoch + 1, config['epochs'] + 1):
                graph, stats = train_epoch(model, optimizer, data, config, epoch, graph, device)
                log(f'Epoch {epoch}: evaluate all {len(data.features):,} rows with fixed corruption')
                rows, predictions = evaluate(model, data, evaluation_mask, graph, config, epoch, device)
                training.append(stats)
                history.extend(rows)
                write_metrics(output, history, predictions, data)
                write_json(output / 'training.json', training)
                checkpoint = dict(epoch=epoch, config=config, dataset=data.manifest,
                                  source_sha256=manifest['source_sha256'], model=model.state_dict(),
                                  optimizer=optimizer.state_dict(), graph=tuple(t.cpu() for t in graph),
                                  metrics=history, training=training, rng=rng_state())
                temporary = output / 'checkpoint.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(output / 'checkpoint.pt')
                del checkpoint
                manifest['last_epoch'] = epoch
                write_json(output / 'run.json', manifest)
        manifest.update(status='complete', elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest['peak_process_rss_gib'] = peak / (2**30 if platform.system() == 'Darwin' else 2**20)
        if device.type == 'cuda':
            manifest['peak_cuda_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
        write_json(output / 'run.json', manifest)
    log(f'Complete: {output}')
