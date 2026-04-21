from pathlib import Path
from typing import List

import numpy as np


def _store_array_result(
    results: dict,
    key: str,
    arr: np.ndarray,
    checkpoint_dir: str | None = None,
    mmap_mode: str | None = "r",
):
    """Store an array in results, optionally checkpointing to disk first."""
    if checkpoint_dir is None:
        results[key] = arr
        return arr

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{key}.npy"
    np.save(target, arr, allow_pickle=False)
    mmapped = np.load(target, mmap_mode=mmap_mode, allow_pickle=False)
    results[key] = mmapped
    return mmapped

def load_results(prefix, dir = Path.cwd(), exclude="", mmap = True):
    results = {}
    for f in dir.iterdir():
        if f.name.startswith(prefix):
            if exclude == "":  
                if mmap:
                    results[f.stem.replace(prefix + "-", "")] = np.load(f, mmap_mode="r", allow_pickle=True)
                else:
                    results[f.stem.replace(prefix + "-", "")] = np.load(f, allow_pickle=True)
            else:
                if exclude not in f.name: 
                    if mmap:
                        results[f.stem.replace(prefix + "-", "")] = np.load(f, mmap_mode="r", allow_pickle=True)
                    else:
                        results[f.stem.replace(prefix + "-", "")] = np.load(f, allow_pickle=True)
    return results
    

def mask_out_events(
    data: np.ndarray,
    results: dict = None,
    event_indices: List[int] | None = None,
    good_indices: np.ndarray | None = None,
    lazy: bool = False,
):
    event_indices = event_indices or []
    if good_indices is None:
        good_indices = np.arange(data.shape[0])
    evt_mask = np.ones(data.shape[0], dtype=bool)
    evt_mask[event_indices] = False
    masked_indices = good_indices[evt_mask]
    if lazy:
        valid_event_indices = np.where(evt_mask)[0]
        if results:
            results["valid_event_indices"] = valid_event_indices
            results["masked_indices"] = masked_indices
        return valid_event_indices, masked_indices
    masked_events = data[evt_mask]
    if results:
        results["masked_events"] = masked_events
        results["masked_indices"] = masked_indices
    return masked_events, masked_indices


def separate_signal_and_noise(
    data,
    results,
    signal_threshold,
    trig_threshold,
    is_sig_rising_edge,
    is_trig_rising_edge,
    source_key=None,
    in_place=True,
    store_waveforms: bool = True,
    checkpoint_dir: str | None = None,
    mmap_mode: str | None = "r",
):
    analysis_data = None
    if source_key and results.get(source_key) is not None:
        analysis_data = results[source_key] if in_place else results[source_key].copy()
    else:
        analysis_data = data if in_place else data.copy()
    n_sig_ch = 7
    n_events = analysis_data.shape[0]

    # Compute masks channel-by-channel to avoid materialising (N, C, S) boolean arrays.
    signal_mask = np.zeros((n_events, n_sig_ch), dtype=bool)
    all_finite_sig = np.ones((n_events, n_sig_ch), dtype=bool)
    for ch in range(n_sig_ch):
        col = analysis_data[:, ch, :]                       # (N, S) view
        finite = np.isfinite(col)
        all_finite_sig[:, ch] = finite.all(axis=1)
        if is_sig_rising_edge:
            signal_mask[:, ch] = (np.fmax.reduce(col, axis=1) > signal_threshold)
        else:
            signal_mask[:, ch] = (np.fmin.reduce(col, axis=1) < signal_threshold)

    noise_mask = ~signal_mask & all_finite_sig
    signal_events, signal_channels = np.where(signal_mask)
    noise_events, noise_channels   = np.where(noise_mask)

    trig_col = analysis_data[:, 7, :]                       # (N, S) view
    trig_finite = np.isfinite(trig_col)
    if is_trig_rising_edge:
        trig_fired = np.fmax.reduce(trig_col, axis=1) > trig_threshold
    else:
        trig_fired = np.fmin.reduce(trig_col, axis=1) < trig_threshold
    trig_signal_mask = trig_fired[:, None]
    trig_noise_mask  = (~trig_fired & trig_finite.all(axis=1))[:, None]
    trig_signal_events, trig_signal_channels = np.where(trig_signal_mask)
    trig_noise_events,  trig_noise_channels  = np.where(trig_noise_mask)
    results["signal_events"] =        signal_events
    results["signal_channels"] =      signal_channels
    results["noise_events"] =         noise_events
    results["noise_channels"] =       noise_channels
    results["trig_signal_events"] =   trig_signal_events
    results["trig_signal_channels"] = trig_signal_channels
    results["trig_noise_events"] =    trig_noise_events
    results["trig_noise_channels"] =  trig_noise_channels
    noise_data = None
    signal_data = None
    trig_signal_data = None
    trig_noise_data = None
    if store_waveforms:
        noise_data = analysis_data[noise_events, noise_channels, :]
        signal_data = analysis_data[signal_events, signal_channels, :]
        trig_signal_data = analysis_data[trig_signal_events, trig_signal_channels, :]
        trig_noise_data = analysis_data[trig_noise_events, trig_noise_channels, :]

        _store_array_result(
            results,
            "noise_data",
            noise_data,
            checkpoint_dir=checkpoint_dir,
            mmap_mode=mmap_mode,
        )
        _store_array_result(
            results,
            "signal_data",
            signal_data,
            checkpoint_dir=checkpoint_dir,
            mmap_mode=mmap_mode,
        )
        _store_array_result(
            results,
            "trig_signal_data",
            trig_signal_data,
            checkpoint_dir=checkpoint_dir,
            mmap_mode=mmap_mode,
        )
        _store_array_result(
            results,
            "trig_noise_data",
            trig_noise_data,
            checkpoint_dir=checkpoint_dir,
            mmap_mode=mmap_mode,
        )
    return signal_events, signal_channels, noise_events, noise_channels, trig_signal_events, trig_signal_channels, trig_noise_events, trig_signal_channels, trig_noise_events, trig_noise_channels, noise_data, signal_data, trig_signal_data, trig_noise_data
