import numpy as np
from typing import Optional, Tuple


def find_peaks(
    data: np.ndarray,
    results: dict,
    source_key: Optional[str] = None,
    in_place: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel min and max peaks across all events.

    Args:
        data (np.ndarray): Shape (events, channels, samples).
        source_key (str, optional): Key in results to use instead of data.
        in_place (bool): Whether to use data directly or copy it.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (channel_min_peaks, channel_max_peaks),
        each of shape (channels, events).
    """
    if source_key and results.get(source_key) is not None:
        analysis_data = results[source_key] if in_place else results[source_key].copy()
    else:
        analysis_data = data if in_place else data.copy()

    # Loop over channels so peak memory stays at O(E*S) instead of O(C*E*S).
    n_ch = analysis_data.shape[1]
    channel_min_peaks = np.empty((n_ch, analysis_data.shape[0]), dtype=float)
    channel_max_peaks = np.empty((n_ch, analysis_data.shape[0]), dtype=float)
    for ch in range(n_ch):
        ch_data = analysis_data[:, ch, :]          # (E, S) view
        channel_min_peaks[ch] = np.fmin.reduce(ch_data, axis=1)
        channel_max_peaks[ch] = np.fmax.reduce(ch_data, axis=1)

    results["channel_min_peaks"] = channel_min_peaks
    results["channel_max_peaks"] = channel_max_peaks
    return channel_min_peaks, channel_max_peaks


def calculate_thresholds(
    data: np.ndarray,
    results: dict,
    percentage_threshold: float | np.ndarray,
    baselines: np.ndarray | str,
    channel_min_peaks: np.ndarray | str,
    channel_max_peaks: np.ndarray | str,
    *,
    edge: str | np.ndarray = "falling",
    in_place=True
) -> np.ndarray:
    """
    Determine y-values to use as the crossing thresholds.

    Args:
        percentage_threshold (float | np.ndarray): The fraction of the channel minimum or maximum that must be
        crossed to be used as threshold value. Can be a single value or set per channel.
        baselines (np.ndarray | str): Shape (channels, events) or a results key to look up.
        channel_min_peaks (np.ndarray | str): Shape (channels, events) or a results key to look up.
        channel_max_peaks (np.ndarray | str): Shape (channels, events) or a results key to look up.
        edge (str | np.ndarray, optional): Determines if a rising or falling edge should be used to determine the
        crossing. Can be set with a single string for all channels or as a numpy array of strings per channel.
        Defaults to "falling".

    Raises:
        ValueError: Mismatch in the shapes of baselines, channel_min_peaks, and channel_max_peaks
        ValueError: percentage_threshold is not a single value or does not match the number of
        channels provided
        ValueError: edge is not a single value or does not match the number of channels provided
        ValueError: Invalid value provided for edge

    Returns:
        np.ndarray: A numpy array of dimensions (channels, events) containing the threshold y-value of each channel
        for a given event.
    """
    # Support results-key lookups for array arguments
    if isinstance(baselines, str):
        baselines = results[baselines]
    if isinstance(channel_min_peaks, str):
        channel_min_peaks = results[channel_min_peaks]
    if isinstance(channel_max_peaks, str):
        channel_max_peaks = results[channel_max_peaks]

    baselines = np.asarray(baselines)
    channel_min_peaks = np.asarray(channel_min_peaks)
    channel_max_peaks = np.asarray(channel_max_peaks)

    if (
        baselines.shape != channel_min_peaks.shape
        or baselines.shape != channel_max_peaks.shape
    ):
        raise ValueError(
            "baselines, channel_min_peaks, channel_max_peaks must all have the same shape "
            f"(got {baselines.shape}, {channel_min_peaks.shape}, {channel_max_peaks.shape})"
        )

    c, e = baselines.shape

    p = np.asarray(percentage_threshold, dtype=float)
    if p.ndim == 0:
        pass
    elif p.shape == (c,):
        p = p[:, None]
    else:
        raise ValueError(
            "percentage_threshold must be scalar or (channels,) " f"got shape {p.shape}"
        )

    if isinstance(edge, str):
        edge_arr = np.full((c,), edge, dtype=object)
    else:
        edge_arr = np.asarray(edge, dtype=object)
        if edge_arr.shape != (c,):
            raise ValueError(
                f"edge must be a string or shape (channels,); got shape {edge_arr.shape}"
            )

    edge_arr = np.char.lower(edge_arr.astype(str))
    is_falling = edge_arr == "falling"
    is_rising = edge_arr == "rising"
    if not np.all(is_falling | is_rising):
        bad = edge_arr[~(is_falling | is_rising)]
        raise ValueError(f'edge entries must be "falling" or "rising"; got {bad!r}')

    thresholds_falling = channel_min_peaks + p * (baselines - channel_min_peaks)
    thresholds_rising = baselines + p * (channel_max_peaks - baselines)

    thresholds = np.where(is_falling[:, None], thresholds_falling, thresholds_rising)
    results["thresholds"] = thresholds
    return thresholds


def find_threshold_x(
    data: np.ndarray,
    results: dict,
    source_key: Optional[str] = None,
    thresholds: np.ndarray = [],
    edge: str | np.ndarray = "falling",
    *,
    falling_threshold: float | np.ndarray | None = None,
    rising_threshold: float | np.ndarray | None = None,
    in_place: bool = True
) -> np.ndarray:
    """Determines the sample indices at which an event crosses its threshold y-value.

    Args:
        data (np.ndarray): A numpy array of dimensions (events, channels, samples).
        thresholds (np.ndarray): Shape (events, channels, samples) or empty to auto-load from
        results["thresholds"] (which has shape (channels, events) and will be broadcast to match data).
        edge (str | np.ndarray, optional): Determines if a rising or falling edge should be used to determine the
        crossing. Can be set with a single string for all channels or as a numpy array of strings per channel.
        falling_threshold (float | np.ndarray | None, optional): Sets the maximum threshold y-value that must be crossed
        for any channels set for falling edges. Defaults to None.
        rising_threshold (float | np.ndarray | None, optional): Sets the minimum threshold y-value that must be crossed
        for any channels set for rising edges. Defaults to None.

    Raises:
        ValueError: Mismatch in shapes of data and thresholds
        ValueError: edge is not a single value or does not match the number of channels provided
        ValueError: falling_threshold is not a single value or does not match the number of channels provided
        ValueError: rising_threshold is not a single value or does not match the number of channels provided

    Returns:
        np.ndarray: A numpy array of dimensions (events, channels) with the threshold x-value for each channel
        in a given event.
    """

    analysis_data = None

    if source_key and results.get(source_key) is not None:
        analysis_data = results[source_key] if in_place else results[source_key].copy()
    else:
        analysis_data = data if in_place else data.copy()

    e, c, s = analysis_data.shape
    thresholds = np.asarray(thresholds)

    # Auto-load and keep as (E, C) — broadcast over samples to avoid (E, C, S) allocation.
    if not thresholds.size:
        if "thresholds" not in results:
            raise ValueError(
                "thresholds not provided and 'thresholds' not found in results"
            )
        thresholds_2d = results["thresholds"].T  # (E, C)
    elif thresholds.shape == (e, c, s):
        thresholds_2d = thresholds[:, :, 0]      # threshold is constant per (E, C)
    else:
        raise ValueError(
            f"data and thresholds must match shape; got {data.shape} vs {thresholds.shape}"
        )

    if isinstance(edge, str):
        edge_arr = np.full((c,), edge, dtype=object)
    else:
        edge_arr = np.asarray(edge, dtype=object)
        if edge_arr.shape != (c,):
            raise ValueError(
                f"edge must be a string or shape (channels,); got {edge_arr.shape}"
            )

    edge_arr = np.char.lower(edge_arr.astype(str))
    is_falling = edge_arr == "falling"
    is_rising  = edge_arr == "rising"
    if not np.all(is_falling | is_rising):
        bad = edge_arr[~(is_falling | is_rising)]
        raise ValueError(f'edge entries must be "falling" or "rising"; got {bad!r}')

    # Validate amplitude threshold shapes up front
    ft_arr = rt_arr = None
    if falling_threshold is not None:
        ft_arr = np.asarray(falling_threshold, dtype=float)
        if ft_arr.ndim not in (0,) and ft_arr.shape != (c,):
            raise ValueError("falling_threshold must be scalar or (channels,)")
    if rising_threshold is not None:
        rt_arr = np.asarray(rising_threshold, dtype=float)
        if rt_arr.ndim not in (0,) and rt_arr.shape != (c,):
            raise ValueError("rising_threshold must be scalar or (channels,)")

    threshold_x = np.full((e, c), np.nan, dtype=float)

    # Process one channel at a time to keep peak memory at O(E*S) instead of O(E*C*S).
    for ch in range(c):
        ch_data = analysis_data[:, ch, :]          # (E, S) — view, no copy
        th      = thresholds_2d[:, ch].astype(float)  # (E,)

        diff    = ch_data.astype(float) - th[:, None]  # (E, S)
        crossed = diff < 0 if is_falling[ch] else diff > 0  # (E, S) bool

        any_cross = crossed.any(axis=1)             # (E,)
        x2        = np.argmax(crossed, axis=1)      # (E,)
        x1        = np.maximum(x2 - 1, 0)

        amp_ok = np.ones(e, dtype=bool)
        if is_falling[ch] and ft_arr is not None:
            ft_ch  = float(ft_arr) if ft_arr.ndim == 0 else float(ft_arr[ch])
            amp_ok = np.fmin.reduce(ch_data, axis=1) <= ft_ch
        if is_rising[ch] and rt_arr is not None:
            rt_ch  = float(rt_arr) if rt_arr.ndim == 0 else float(rt_arr[ch])
            amp_ok = np.fmax.reduce(ch_data, axis=1) >= rt_ch

        valid  = any_cross & (x2 > 0) & amp_ok
        ev_ok  = np.where(valid)[0]
        if ev_ok.size == 0:
            continue

        y2    = ch_data[ev_ok, x2[ev_ok]].astype(float)
        y1    = ch_data[ev_ok, x1[ev_ok]].astype(float)
        t_val = th[ev_ok]
        denom = y2 - y1
        ok    = np.isfinite(denom) & (denom != 0)

        threshold_x[ev_ok[ok], ch] = (
            (t_val[ok] - y1[ok]) * (x2[ev_ok[ok]] - x1[ev_ok[ok]]) / denom[ok]
            + x1[ev_ok[ok]]
        )

    results["threshold_x"] = threshold_x
    return threshold_x
