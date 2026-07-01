"""Optional GPU-memory housekeeping for long runs.

A long GCMC/RE run relaxes structures whose atom count keeps changing, so the
PyTorch CUDA caching allocator reserves a fresh block-set for every new size and
never returns it to the driver. Reserved memory then drifts far above the live
footprint and can drive spurious OOM. The preferred fix is to launch with
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``; ``empty_cuda_cache`` is an
in-loop fallback that returns the pooled blocks to the driver.

Both helpers are safe no-ops when torch is absent or no CUDA device is present,
so importing them never forces a torch dependency (CI runs without torch).
"""


def empty_cuda_cache() -> bool:
    """Release cached-but-unused CUDA blocks. Returns True if a cache was
    actually emptied, False when torch/CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return True
    return False


def should_empty_cache(step: int, interval: int) -> bool:
    """True on the steps where a periodic cache flush is due. ``interval`` <= 0
    disables it; the flush never fires on step 0."""
    return interval > 0 and step > 0 and step % interval == 0
