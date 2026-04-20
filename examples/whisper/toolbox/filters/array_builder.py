from . import templates as ft
from .data_forming_filters import PruneFilter
import numpy as np
from tqdm import tqdm
import tempfile
import os
import shutil
from pathlib import Path


FILL_CHUNK = 10_000


def _make_memmap(tmpdir, name, dtype, shape, fill_value):
    """Create a memory-mapped array backed by a temp file, filled in chunks."""
    path = os.path.join(tmpdir, f"{name}.dat")
    mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    n = shape[0]
    for start in range(0, n, FILL_CHUNK):
        end = min(start + FILL_CHUNK, n)
        mm[start:end] = fill_value
    return mm


class ArrayBuilder(ft.Filter):
    """Stage 3 of FET: writes events into preallocated (N, C, S) arrays.

    Accumulates rows by side effect inside _process; the event is passed
    through unchanged per the Filter contract. Call to_dict() after
    iteration to retrieve the output.

    Uses memory-mapped temp files so peak RAM is bounded by the OS page
    cache rather than total array size.

    Args:
        n_events:    Number of events to preallocate for.
        channels:    ASIC channel indices to extract. Must be non-empty.
        max_samples: Samples per channel per event. Must be a positive
                     multiple of 64.
        pedestals:   Optional pedestal array (n_ch, n_windows, 64).
                     If provided, subtracts pedestals[ch][wl] inline.
    """

    def __init__(self, n_events: int, channels: list[int], max_samples: int,
                 pedestals: np.ndarray | None = None):
        super().__init__(in_place=True)
        m_samples_per_win = 32
        if max_samples <= 0 or max_samples % m_samples_per_win != 0:
            raise ValueError(f"max_samples must be a positive multiple of 64, got {max_samples}")
        if not channels:
            raise ValueError("channels must be a non-empty list")
        n_ch = len(channels)
        n_windows = max_samples // m_samples_per_win
        self.channels = channels
        self.pedestals = pedestals
        self.channel_map = {ch: col for col, ch in enumerate(channels)}
        self._max_samples = max_samples
        self._n_windows = n_windows
        self._row = 0
        self._tmpdir = tempfile.mkdtemp(prefix="arraybuilder_", dir = Path().cwd())
        tqdm.write(f"ArrayBuilder tmpdir: {self._tmpdir}")
        self.raw   = _make_memmap(self._tmpdir, "raw",   np.float32, (n_events, n_ch, max_samples), np.nan)
        self.data   = _make_memmap(self._tmpdir, "data",   np.float32, (n_events, n_ch, max_samples), np.nan)
        self.wl     = _make_memmap(self._tmpdir, "wl",     np.uint16,  (n_events, n_ch, n_windows),   0xFFFF)
        self.timing = _make_memmap(self._tmpdir, "timing", np.float32, (n_events, n_ch, n_windows),   np.nan)

    def _process(self, event) -> dict | None:
        row = self._row
        self._row += 1
        for col, ch in enumerate(self.channels):
            raw = np.array(event["data"][ch], dtype=np.float32)
            wl = np.array(event["window_labels"][ch], dtype=np.uint16)
            timing = np.array(event["timing"][ch], dtype=np.float32) if np.all(event.get("timing")) else None
            s = min(len(raw), self._max_samples)
            w = min(len(wl),  self._n_windows)
            self.raw[row, col, :s] = raw[:s]
            if self.pedestals is not None:
                raw = raw - self.pedestals[ch][wl].flatten().astype(np.float32) 
            self.data[row, col, :s] = raw[:s]
            self.wl[row, col, :w] = wl[:w]
            self.timing[row, col, :w] = timing[:w] if np.all(event.get("timing")) else np.repeat(np.nan, w)
        return event

    def to_dict(self, good_indices: list[int]) -> dict:
        n = self._row
        result = {
            "raw":           self.raw[:n],
            "data":          self.data[:n],
            "window_labels": self.wl[:n],
            "timing":        self.timing[:n],
            "channels":      np.array(self.channels, dtype=int),
            "channel_map":   self.channel_map,
            "good_indices":  np.array(good_indices[:n], dtype=np.uint32),
        }
        # Do NOT call _cleanup() here — the memmap slices above still reference
        # the backing files. Cleanup happens via __del__ when this builder is GC'd.
        return result

    def _cleanup(self):
        for attr in ("raw", "data", "wl", "timing"):
            mm = getattr(self, attr, None)
            if mm is not None:
                del mm
            setattr(self, attr, None)
        if hasattr(self, "_tmpdir") and self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __del__(self):
        self._cleanup()
