"""Declarative analysis pipeline with dependency validation.

Step functions live in their source modules; this file provides the scaffold
(Step metadata + AnalysisPipeline runner) and pre-defined step instances.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm


class MmapDict(dict):
    """Dict that transparently stores numpy arrays as memory-mapped files."""

    def __init__(self, tmpdir=None):
        super().__init__()
        self._tmpdir = tmpdir or str(Path.cwd() / "tmp")
        os.makedirs(self._tmpdir, exist_ok=True)
        self._paths = {}

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            path = os.path.join(self._tmpdir, f"{key}.npy")
            fp = np.lib.format.open_memmap(path, mode='w+', dtype=value.dtype, shape=value.shape)
            fp[:] = value
            super().__setitem__(key, fp)
            self._paths[key] = path
        else:
            super().__setitem__(key, value)

    def cleanup(self):
        for path in self._paths.values():
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class Step:
    """Metadata wrapper around a step function.

    Attributes:
        name: human-readable identifier
        fn: callable with signature (data, results, **params) -> None
        requires: keys that must exist in results before this step runs
        produces: maximum set of keys this step may write to results
        index_keys: subset of produces whose values are event-index arrays
            (1-D integer arrays or (K, 2) event-channel pairs).  In chunked
            mode these are offset by the chunk start before concatenation.
    """

    name: str
    fn: Callable[..., None]
    requires: frozenset[str]
    produces: frozenset[str]
    index_keys: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Helpers for chunked accumulation
# ---------------------------------------------------------------------------

def _detect_mode(key: str, val, chunk_n: int, index_keys: frozenset) -> str:
    """Determine how to accumulate *val* across chunks.

    Modes
    -----
    'axis0'      shape[0] == chunk_n  → pre-allocate full (N, ...) mmap, fill per chunk
    'axis_last'  shape[-1] == chunk_n and shape[0] < chunk_n
                 → pre-allocate full (..., N) mmap, fill per chunk
    'index'      key is in index_keys → integer event-index array, offset + concat
    'event_dict' dict with all-integer keys → event-keyed dict, offset keys + merge
    'concat'     anything else that is an ndarray → concatenate on axis 0 as-is
    'scalar'     non-array, non-dict → keep last chunk's value
    """
    if key in index_keys:
        return 'index'
    if isinstance(val, dict):
        if val and all(isinstance(k, (int, np.integer)) for k in val):
            return 'event_dict'
        return 'scalar'
    if not isinstance(val, np.ndarray):
        return 'scalar'
    if val.ndim >= 1 and val.shape[0] == chunk_n:
        return 'axis0'
    if val.ndim >= 2 and val.shape[-1] == chunk_n and val.shape[0] < chunk_n:
        return 'axis_last'
    return 'concat'


def _combine(mode: str, chunks: list[tuple[int, object]]) -> object:
    """Combine per-chunk values according to *mode*.

    Parameters
    ----------
    mode:   one of the mode strings returned by _detect_mode
    chunks: list of (chunk_start, value) pairs
    """
    if mode == 'scalar':
        return chunks[-1][1]

    if mode == 'event_dict':
        merged: dict = {}
        for start, d in chunks:
            for event_idx, v in d.items():
                merged[int(event_idx) + start] = v
        return merged

    arrays = [(s, v) for s, v in chunks if isinstance(v, np.ndarray)]
    if not arrays:
        return chunks[-1][1]

    if mode in ('axis0', 'concat'):
        return np.concatenate([v for _, v in arrays], axis=0)

    if mode == 'axis_last':
        return np.concatenate([v for _, v in arrays], axis=-1)

    if mode == 'index':
        parts = []
        for start, v in arrays:
            if v.ndim == 1:
                parts.append(v + start)
            elif v.ndim == 2:
                c = v.copy()
                c[:, 0] = c[:, 0] + start
                parts.append(c)
            else:
                parts.append(v)
        return np.concatenate(parts, axis=0) if parts else arrays[-1][1]

    return chunks[-1][1]


# ---------------------------------------------------------------------------
# AnalysisPipeline
# ---------------------------------------------------------------------------

class AnalysisPipeline:
    """Builder for chaining analysis steps with dependency validation.

    Usage (non-chunked)::

        pipe = AnalysisPipeline()
        pipe.add(baseline_step, interval=(0, 100))
        pipe.add(peak_step, search_start=100, find_min=True)
        pipe.add(cfd_step, fraction=0.5)
        results = pipe.run(data)

    Usage (chunked, for datasets that don't fit in RAM)::

        results = pipe.run(data, chunk_size=5_000, tmpdir="/data/tmp")

    With ``chunk_size`` set each step processes ``chunk_size`` events at a
    time, keeping peak RAM proportional to the chunk rather than the full
    dataset.  Per-event arrays (shape ``(N, ...)`` or ``(C, N)``) are written
    incrementally into pre-allocated memory-mapped files so they never need to
    be fully materialised in RAM.  Sparse index / timing arrays are
    concatenated after all chunks finish.

    Notes
    -----
    - In chunked mode the data array is **always copied** per chunk, so
      in-place step functions cannot corrupt the source mmap.
    - ``copy_on_write`` and ``chunk_size`` are mutually exclusive.
    - Steps that produce aggregate statistics across all events
      (signal-to-signal pairings, channel-pair counts, etc.) see only the
      current chunk in chunked mode.  Run those steps separately on the
      full mmap'd output if you need global aggregates.
    """

    def __init__(self) -> None:
        self._steps: list[tuple[Step, dict]] = []

    def add(self, step: Step, **params) -> AnalysisPipeline:
        """Append a step with its parameters. Returns self for chaining."""
        self._steps.append((step, params))
        return self

    def validate(self) -> None:
        """Check that every step's requires are satisfied by prior produces.

        Raises:
            ValueError: if a step's requirements are not met.
        """
        available: set[str] = set()
        for step, _ in self._steps:
            missing = step.requires - available
            if missing:
                raise ValueError(
                    f"Step '{step.name}' requires {missing} "
                    f"but only {available or '{}'} available from prior steps"
                )
            available |= step.produces

    def run(
        self,
        data: np.ndarray,
        chunk_size: int | None = None,
        tmpdir: str | None = None,
        copy_on_write: bool = False,
    ) -> dict:
        """Validate then execute all steps, returning the accumulated results.

        Parameters
        ----------
        data:
            Input array of shape ``(events, channels, samples)``.
        chunk_size:
            If given, process this many events at a time.  Results for
            per-event arrays are written incrementally into pre-allocated
            memory-mapped files under *tmpdir*.  Cannot be combined with
            ``copy_on_write=True``.
        tmpdir:
            Directory for temporary / output mmap files when
            ``chunk_size`` is set.  A fresh system temp dir is created when
            ``None``.  Pass a persistent path to keep results on disk after
            the call returns.
        copy_on_write:
            If True, reopen the source memmap with mode='c' so steps can
            modify it in-place without touching the file on disk.  Requires
            *data* to be a ``np.memmap``.  Incompatible with ``chunk_size``.
        """
        self.validate()

        if chunk_size is not None and copy_on_write:
            raise ValueError("chunk_size and copy_on_write cannot both be set")

        if copy_on_write:
            if not isinstance(data, np.memmap):
                raise ValueError("copy_on_write=True requires data to be a numpy memmap")
            data = np.memmap(data.filename, dtype=data.dtype, mode='c', shape=data.shape)

        if chunk_size is not None:
            return self._run_chunked(data, chunk_size, tmpdir)

        results: dict = MmapDict()
        for step, params in tqdm(self._steps, desc="Running analysis pipeline", unit="step"):
            step.fn(data, results, **params)
        return results

    def _run_chunked(self, data: np.ndarray, chunk_size: int, tmpdir: str | None) -> dict:
        """Process data in chunks to bound peak RAM usage.

        Per-event output arrays are pre-allocated as memory-mapped files of
        full size (N, ...) or (C, N) and filled one chunk at a time.  Index
        and sparse arrays are accumulated in RAM and concatenated at the end
        (they are much smaller than the waveform data).
        """
        if tmpdir is None:
            tmpdir = Path.cwd() / "tmp"
        os.makedirs(tmpdir, exist_ok=True)

        N = data.shape[0]
        all_index_keys: frozenset[str] = frozenset().union(
            *(s.index_keys for s, _ in self._steps)
        )

        modes: dict[str, str] = {}
        pre_alloc: dict[str, np.memmap] = {}
        concat_accum: dict[str, list[tuple[int, object]]] = {}

        for chunk_idx, start in enumerate(
            tqdm(range(0, N, chunk_size), desc="Running analysis pipeline (chunked)", unit="chunk")
        ):
            end = min(start + chunk_size, N)
            actual_n = end - start
            chunk = data[start:end].copy()

            chunk_results: dict = {}
            for step, params in self._steps:
                step.fn(chunk, chunk_results, **params)

            for key, val in chunk_results.items():
                if chunk_idx == 0:
                    mode = _detect_mode(key, val, actual_n, all_index_keys)
                    modes[key] = mode

                    if mode == 'axis0' and isinstance(val, np.ndarray):
                        full_shape = (N, *val.shape[1:])
                        path = os.path.join(tmpdir, f"{key}.npy")
                        fp = np.lib.format.open_memmap(
                            path, mode='w+', dtype=val.dtype, shape=full_shape
                        )
                        pre_alloc[key] = fp

                    elif mode == 'axis_last' and isinstance(val, np.ndarray):
                        full_shape = (*val.shape[:-1], N)
                        path = os.path.join(tmpdir, f"{key}.npy")
                        fp = np.lib.format.open_memmap(
                            path, mode='w+', dtype=val.dtype, shape=full_shape
                        )
                        pre_alloc[key] = fp

                if key in pre_alloc:
                    if modes[key] == 'axis0':
                        pre_alloc[key][start:end] = val
                    else:
                        pre_alloc[key][..., start:end] = val
                else:
                    if key not in concat_accum:
                        concat_accum[key] = []
                        if key not in modes:
                            modes[key] = _detect_mode(key, val, actual_n, all_index_keys)
                    concat_accum[key].append((start, val))

        results: dict = dict(pre_alloc)
        for key, chunks in concat_accum.items():
            mode = modes.get(key, 'concat')
            results[key] = _combine(mode, chunks)

        return results

    def print_keys(self, results: dict, end="\n"):
        for key in results.keys():
            print(key, end=end)

    def save_results(
        self,
        results: dict,
        save_name: str,
        save_dir: str = "",
        keys: list[str] | None = None,
        mmap_mode: str | None = None,
        drop_in_memory: bool = False,
    ) -> dict[str, Path]:
        """Save result keys to per-key .npy files.

        Args:
            results: Result dictionary produced by run().
            save_name: Prefix used for filenames (e.g., "chip0").
            save_dir: Directory where files are written.
            keys: Optional subset of result keys to save; defaults to all keys.
            mmap_mode: If set (e.g., "r"), replace array values in results with
                memmapped arrays loaded from disk.
            drop_in_memory: If True and mmap_mode is None, replace saved keys in
                results with file path strings.

        Returns:
            Mapping from result key to file path.
        """
        target_dir = Path(save_dir) if save_dir else Path(".")
        target_dir.mkdir(parents=True, exist_ok=True)
        to_save = keys if keys is not None else list(results.keys())
        saved: dict[str, Path] = {}

        print("Saving keys")
        for key in tqdm(to_save, desc="Saving results"):
            if key not in results:
                continue
            value = results[key]
            out_path = target_dir / f"{save_name}-{key}.npy"
            np.save(out_path, value, allow_pickle=True)
            saved[key] = out_path
            can_memmap = (
                mmap_mode is not None
                and isinstance(value, np.ndarray)
                and not value.dtype.hasobject
            )
            if can_memmap:
                results[key] = np.load(out_path, mmap_mode=mmap_mode, allow_pickle=True)
            elif drop_in_memory:
                results[key] = str(out_path)
        return saved

    def __repr__(self) -> str:
        lines = [f"AnalysisPipeline({len(self._steps)} steps):"]
        for i, (step, params) in enumerate(self._steps):
            param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
            lines.append(f"  {i}: {step.name}({param_str})")
        return "\n".join(lines)
