from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple

from enum import Enum




def find_baseline_samples(
    data: np.ndarray,
    window_len: int,
    step: int = 1,
) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"data must be (events, channels, samples); got shape {data.shape}")

    n_events, n_channels, n_samples = data.shape

    if not (1 <= window_len <= n_samples):
        raise ValueError(f"window_len must be in [1, {n_samples}], got {window_len}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    n_windows = (n_samples - window_len) // step + 1

    var_x = np.full((n_events, n_channels, n_windows), np.inf)
    for w in range(n_windows):
        start = w * step
        win = data[:, :, start:start + window_len]
        valid_count = np.sum(~np.isnan(win), axis=2)
        full = valid_count >= window_len
        v = np.nanvar(win.astype(np.float64, copy=False), axis=2)
        var_x[:, :, w] = np.where(full, v, np.inf)

    has_any_valid = np.isfinite(var_x).any(axis=2)
    best_win_idx  = np.argmin(var_x, axis=2)
    best_start    = best_win_idx * step

    s = np.arange(n_samples)
    in_window = (
        (s[None, None, :] >= best_start[:, :, None]) &
        (s[None, None, :] <  best_start[:, :, None] + window_len) &
        has_any_valid[:, :, None]
    )
    masked = np.where(in_window, data, np.nan)

    return masked, best_start

class BaselineMethod(Enum):
    """
    Valid methods for the Baseline class for multiple events using waveform data

    Values
    ------
    SLIDING_WINDOW
        Uses a sliding window to determine which set of samples in a channel data should be used
        to determine the baseline
    FIXED_WINDOW
        Divides channel data into fixed (not sliding) windows and determines which windows should
        be used to determine the baseline
    NTH_WINDOW
        Only use samples in the NTH window to calculate the baseline
    """
    SLIDING_WINDOW = 0
    FIXED_WINDOW = 1
    NTH_WINDOW = 2

class BaselineAxes(Enum):
    """
    Valid methods for the Baseline class for multiple events using waveform data

    Values
    ------
    SLIDING_WINDOW
        Uses a sliding window to determine which set of samples in a channel data should be used
        to determine the baseline
    FIXED_WINDOW
        Divides channel data into fixed (not sliding) windows and determines which windows should
        be used to determine the baseline
    NTH_WINDOW
        Only use samples in the NTH window to calculate the baseline
    """
    EVENT = 0
    CHANNEL = 1
    SAMPLE = 2

class Baseline():
    """
    Used to calculate the baseline for a series of events. Event data should be pre-processed with 
    filters, pedestals, and other corrections before calculating baselines.
    """
    def __init__(self, method: BaselineMethod, axis: BaselineAxes = BaselineAxes.CHANNEL):
        """
        Selects the method used to calculate the baseline.

        Args:
            method (BaselineMethod): The method used to calculate the baseline
        """
        self.method = method
        self.axis = axis

    def run(self, data: np.ndarray, *, window_len: int, sliding_window_step: int = 1, window_index: int = 0):
        if data.ndim != 3:
            raise ValueError(f"data must be (events, channels, samples); got shape {data.shape}")
        
        if self.method == BaselineMethod.SLIDING_WINDOW:
            baseline_events, baseline_start_idx = self.__sliding_window(data, window_len, sliding_window_step)
            baselines = None

            if self.axis == BaselineAxes.CHANNEL:
                baselines = np.nanmean(baseline_events.swapaxes(0,1), axis=(2))

            return baselines, (baseline_events, baseline_start_idx)
        
        elif self.method == BaselineMethod.FIXED_WINDOW:
            baseline_events, baseline_win_idx = self.__fixed_window(data, window_len)
            baselines = None

            if self.axis == BaselineAxes.CHANNEL:
                baselines = np.nanmean(baseline_events, axis=-1).T  # (N,C) -> (C,N)

            return baselines, (baseline_events, baseline_win_idx)
        elif self.method == BaselineMethod.NTH_WINDOW:
            baseline_events = self.__nth_window(data, window_len, window_index)
            baselines = None

            if self.axis == BaselineAxes.CHANNEL:
                baselines = np.nanmean(baseline_events, axis=(2)).swapaxes(0, 1)
            
            return baselines

        else:
            return np.array([0])
    
    def __sliding_window(self, data: np.ndarray,  window_len: int, step: int = 1) -> (np.ndarray, np.ndarray):
            """
            Uses a sliding window of length window_len to calculate the variance for each window in 
            channel data and determine which window has the lowest variance. Masks out all values
            outside the chosen window with np.nan.

            Args:
                data (np.ndarray): Data with shape (EVENTS, CHANNELS, SAMPLES)
                window_len (int): The size of the sliding window appleid to channel data
                step (int, optional): The interval of sliding windows that are considered to determine
                which windows are used to calcualte the baseline. Defaults to 1.

            Raises:
                ValueError: The number of samples must be greater than or equal to the the length of 
                the sliding window and the length of the sliding window must be at least one. 
                ValueError: The step size must be at least 1. 

            Returns:
                np.ndarray: An array with dimensions (EVENTS, CHANNELS, SAMPLES) with any samples outside
                of the window with the lowest variance masked as np.nan.
                np.ndarray: An array with dimensions (EVENTS, CHANNELs) containing the index of the window
                with the lowest variance.
            """
            n_events, n_channels, n_samples = data.shape

            if not (1 <= window_len <= n_samples):
                raise ValueError(f"window_len must be in [1, {n_samples}], got {window_len}")
            if step < 1:
                raise ValueError(f"step must be >= 1, got {step}")

            n_windows = (n_samples - window_len) // step + 1

            var_x = np.full((n_events, n_channels, n_windows), np.inf)
            for w in range(n_windows):
                start = w * step
                win = data[:, :, start:start + window_len]
                valid_count = np.sum(~np.isnan(win), axis=2)
                full = valid_count >= window_len
                v = np.nanvar(win.astype(np.float64, copy=False), axis=2)
                var_x[:, :, w] = np.where(full, v, np.inf)

            has_any_valid = np.isfinite(var_x).any(axis=2)
            best_win_idx  = np.argmin(var_x, axis=2)
            best_start    = best_win_idx * step

            s = np.arange(n_samples)
            in_window = (
                (s[None, None, :] >= best_start[:, :, None]) &
                (s[None, None, :] <  best_start[:, :, None] + window_len) &
                has_any_valid[:, :, None]
            )
            masked = np.where(in_window, data, np.nan)

            return masked, best_start
    
    def __fixed_window(self, data: np.ndarray, window_length: int) -> (np.ndarray, np.ndarray):
        """
        Uses a non-sliding fixed window length to divide the samples in channel data. Recommended to set to 
        the length of a sampling window.

        Args:
            data (np.ndarray): Data with shape (EVENTS, CHANNELS, SAMPLES)
            window_len (int): The size of the fixed window appleid to channel data

        Returns:
            np.ndarray: An array with dimensions (EVENTS, CHANNELS, SAMPLES) with any samples outside
            of the window with the lowest variance masked as np.nan.
            np.ndarray: An array with dimensions (EVENTS, CHANNELs) containing the index of the window
            with the lowest variance.
        """
        num_events, num_channels, num_samples = data.shape
        windowed_events = data.reshape(num_events, num_channels, num_samples // window_length, window_length)
        windowed_variances = []

        for event_idx, event in enumerate(windowed_events):
            windowed_variances.append(np.nanvar(event, axis=(2), ddof=1))

        windowed_variances = np.array(windowed_variances)

        channel_windowed_variances = windowed_variances.swapaxes(0, 1)

        
        valid = ~np.all(np.isnan(channel_windowed_variances), axis=2) 

        filled = np.where(np.isnan(channel_windowed_variances), np.inf, channel_windowed_variances)

        channel_windowed_min_variance_idx = np.argmin(filled, axis=2) 

        channel_windowed_min_variance_idx = np.where(
            valid,
            channel_windowed_min_variance_idx,
            np.nan,
        )

        windowed_baselines = data.copy()  # (N, C, S)
        for event_idx in range(num_events):
            for channel_idx in range(num_channels):
                if not np.isnan(channel_windowed_min_variance_idx[channel_idx, event_idx]):
                    win_idx = int(channel_windowed_min_variance_idx[channel_idx, event_idx])
                    start = win_idx * window_length
                    end = start + window_length
                    windowed_baselines[event_idx, channel_idx, :start] = np.nan
                    windowed_baselines[event_idx, channel_idx, end:] = np.nan
                else:
                    windowed_baselines[event_idx, channel_idx, :] = np.nan

        return windowed_baselines, channel_windowed_min_variance_idx

    def __nth_window(self, data: np.ndarray, window_len: int, window_index: int):
        nth_window = data[:, :, window_index*window_len:(window_index+1)*window_len]
        return nth_window



def find_baseline_samples(data, results, method, axis=1, window_len=64, sliding_window_step=64, window_index=0, source_key = None, in_place = True):
    if data.ndim != 3:
        raise ValueError(f"data must be (events, channels, samples); got shape {data.shape}")
    
    analysis_data = None 

    if source_key and results.get(source_key) is not None:
        analysis_data = results[source_key] if in_place else results[source_key].copy()
    else:
        analysis_data = data if in_place else data.copy()

    if method == 0:
        baseline_events, baseline_start_idx = __sliding_window(analysis_data, window_len, sliding_window_step)
        baselines = None

        if axis == 1:
            baselines = np.nanmean(baseline_events.swapaxes(0,1), axis=(2))
        results["baselines_sliding_window"] = baselines
        results["baseline_sliding_window_events"] = baseline_events
        results["baseline_sliding_window_start_idx"] = baseline_start_idx
        return baselines, (baseline_events, baseline_start_idx)
    
    elif method == 1:
        baseline_events, baseline_win_idx = __fixed_window(analysis_data, window_len)
        baselines = None

        if axis == 1:
            baselines = np.nanmean(baseline_events, axis=-1).T  # (N,C) -> (C,N)
        results["baselines_fixed_window"] = baselines
        results["baseline_fixed_window_events"] = baseline_events
        results["baseline_fixed_window_win_idx"] = baseline_win_idx
        return baselines, (baseline_events, baseline_win_idx)
    elif method == 2:
        baseline_events = __nth_window(analysis_data, window_len, window_index)
        baselines = None

        if axis == 2:
            baselines = np.nanmean(baseline_events, axis=(2)).swapaxes(0, 1)
        results[f"baselines_{window_index}_window"] = baselines
        return baselines

    else:
        return np.array([0])

def __sliding_window(data: np.ndarray, window_len: int, step: int = 1) -> (np.ndarray, np.ndarray):
    """
    Uses a sliding window of length window_len to calculate the variance for each window in
    channel data and determine which window has the lowest variance. Masks out all values
    outside the chosen window with np.nan.

    Args:
        data (np.ndarray): Data with shape (EVENTS, CHANNELS, SAMPLES)
        window_len (int): The size of the sliding window appleid to channel data
        step (int, optional): The interval of sliding windows that are considered to determine
        which windows are used to calcualte the baseline. Defaults to 1.

    Raises:
        ValueError: The number of samples must be greater than or equal to the the length of
        the sliding window and the length of the sliding window must be at least one.
        ValueError: The step size must be at least 1.

    Returns:
        np.ndarray: An array with dimensions (EVENTS, CHANNELS, SAMPLES) with any samples outside
        of the window with the lowest variance masked as np.nan.
        np.ndarray: An array with dimensions (EVENTS, CHANNELs) containing the index of the window
        with the lowest variance.
    """
    n_events, n_channels, n_samples = data.shape

    if not (1 <= window_len <= n_samples):
        raise ValueError(f"window_len must be in [1, {n_samples}], got {window_len}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    n_windows = (n_samples - window_len) // step + 1

    # Iterate over window positions to avoid materialising an (N, C, W, S) array.
    # Each iteration keeps at most O(N·C·S) live memory instead of O(N·C·W·S).
    var_x = np.full((n_events, n_channels, n_windows), np.inf)
    for w in range(n_windows):
        start = w * step
        win = data[:, :, start:start + window_len]              # view, no copy
        valid_count = np.sum(~np.isnan(win), axis=2)            # (N, C)
        full = valid_count >= window_len
        v = np.nanvar(win.astype(np.float64, copy=False), axis=2)
        var_x[:, :, w] = np.where(full, v, np.inf)

    has_any_valid = np.isfinite(var_x).any(axis=2)              # (N, C)
    best_win_idx  = np.argmin(var_x, axis=2)                   # (N, C)
    best_start    = best_win_idx * step                         # (N, C)

    # Vectorised masking: build in-window boolean mask (N, C, S)
    s = np.arange(n_samples)
    in_window = (
        (s[None, None, :] >= best_start[:, :, None]) &
        (s[None, None, :] <  best_start[:, :, None] + window_len) &
        has_any_valid[:, :, None]
    )
    masked = np.where(in_window, data, np.nan)

    return masked, best_start

def __fixed_window(data: np.ndarray, window_length: int) -> (np.ndarray, np.ndarray):
    """
    Uses a non-sliding fixed window length to divide the samples in channel data. Recommended to set to 
    the length of a sampling window.

    Args:
        data (np.ndarray): Data with shape (EVENTS, CHANNELS, SAMPLES)
        window_len (int): The size of the fixed window appleid to channel data

    Returns:
        np.ndarray: An array with dimensions (EVENTS, CHANNELS, SAMPLES) with any samples outside
        of the window with the lowest variance masked as np.nan.
        np.ndarray: An array with dimensions (EVENTS, CHANNELs) containing the index of the window
        with the lowest variance.
    """
    num_events, num_channels, num_samples = data.shape
    windowed_events = data.reshape(num_events, num_channels, num_samples // window_length, window_length)
    windowed_variances = []

    for event_idx, event in enumerate(windowed_events):
        windowed_variances.append(np.nanvar(event, axis=(2), ddof=1))

    windowed_variances = np.array(windowed_variances)

    channel_windowed_variances = windowed_variances.swapaxes(0, 1)

    
    valid = ~np.all(np.isnan(channel_windowed_variances), axis=2) 

    filled = np.where(np.isnan(channel_windowed_variances), np.inf, channel_windowed_variances)

    channel_windowed_min_variance_idx = np.argmin(filled, axis=2) 

    channel_windowed_min_variance_idx = np.where(
        valid,
        channel_windowed_min_variance_idx,
        np.nan,
    )

    windowed_baselines = data.copy()  # (N, C, S)
    for event_idx in range(num_events):
        for channel_idx in range(num_channels):
            if not np.isnan(channel_windowed_min_variance_idx[channel_idx, event_idx]):
                win_idx = int(channel_windowed_min_variance_idx[channel_idx, event_idx])
                start = win_idx * window_length
                end = start + window_length
                windowed_baselines[event_idx, channel_idx, :start] = np.nan
                windowed_baselines[event_idx, channel_idx, end:] = np.nan
            else:
                windowed_baselines[event_idx, channel_idx, :] = np.nan

    return windowed_baselines, channel_windowed_min_variance_idx

def __nth_window(data: np.ndarray, window_len: int, window_index: int):
    nth_window = data[:, :, window_index*window_len:(window_index+1)*window_len]
    return nth_window