"""Always-visible startup stages, without importing ML dependencies."""

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def startup_stage(label, device=None):
    """Report a stage before work starts and its wall time after completion.

    CUDA/MPS initialization stages synchronize at completion so queued GPU work
    is included in the elapsed time. This is not used inside the epoch loop.
    """
    print(f"[Startup] {label} ...", flush=True)
    started = perf_counter()
    try:
        yield
        if device is not None and device.type == 'cuda':
            import torch
            torch.cuda.synchronize(device)
        elif device is not None and device.type == 'mps':
            import torch
            torch.mps.synchronize()
    except BaseException:
        print(f"[Startup] {label} failed/interrupted ({perf_counter() - started:.2f}s)",
              flush=True)
        raise
    else:
        print(f"[Startup] {label} done ({perf_counter() - started:.2f}s)", flush=True)
