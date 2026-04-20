from .template import Filter, FilterPipeline

import numpy as np
import scipy
from typing import Dict

Event = Dict
class FilterSpikes(Filter):
    """
    Mask out or interpolate any samples that are determined to be digitizer spikes
    based on their value relative to neighboring samples.
    """
    def __init__(self, k: float = 5.0, edge_scale: float=2.6, in_place: bool =True, apply_correction=False, min_diff_threshold: float = 0.0, baseline_lookback_window = 0):
        """Creates a filter for spikes

        Keyword Arguments:
            k -- The scale factor for the relative difference
            of a sample to its neighboring samples (default: {5.0})
            edge_scale -- The scale factor applied to samples
            located at the beginning and end of a channel. (default: {2.6})
            in_place -- Determines if corrections are applied in place.  (default: {True})
            apply_correction -- Determines if a spike is replaced with the average of its neighboring samples (default: {False})
            min_diff_threshold -- Minimum absolute difference from neighbors required to be considered an extreme point (default: {0.0})
            baseline_lookback_window -- The number of indices before and after a sample point that will be used to estimate the median of that window. 
        """
        super().__init__(in_place=in_place)
        self.k = k
        self.edge_scale = edge_scale
        self.apply_correction = apply_correction
        self.min_diff_threshold = min_diff_threshold
        self.baseline_lookback_window = baseline_lookback_window

    def _process(self, event: Event) -> Event | None:
        """Processes an event with the filter

        Arguments:
            event -- An event

        Returns:
            An event with the processed data
        """
        if "data" not in event:
            return event

        event_data = event["data"]
        if not isinstance(event_data, np.ndarray):
            event["data"] = np.array(event_data, dtype=float)
            event_data = event["data"]

        num_channels, n = event_data.shape
        w = max(self.baseline_lookback_window, 1)

        for ch in range(num_channels):
            y = event_data[ch]

            if n < 2:
                continue
            
            # Calculate differences between consecutive samples
            diffs = np.abs(np.diff(y))
            diffs = diffs[np.isfinite(diffs)]

            # Get the median difference between samples
            s = float(np.median(diffs)) if diffs.size else 0.0
            s = max(s, np.finfo(float).eps)

            # Threshold values for how much a sample can differ from the median of the window
            t = self.k * s
            t_edge = self.edge_scale * t

            spike_mask = np.zeros(n, dtype=bool)

            # edges
            if np.isfinite(y[0]) and np.isfinite(y[1]) and abs(y[0] - y[1]) > t_edge:
                spike_mask[0] = True
            if np.isfinite(y[-1]) and np.isfinite(y[-2]) and abs(y[-1] - y[-2]) > t_edge:
                spike_mask[-1] = True

            
            y_nan = np.where(np.isfinite(y), y, np.nan)
            pad = np.pad(y_nan, w, constant_values=np.nan)

            # Split data into sliding window
            shape = (n, 2 * w + 1)
            strides = (pad.strides[0], pad.strides[0])
            windows = np.lib.stride_tricks.as_strided(pad, shape=shape, strides=strides)

            # Calculate median of samples in the sliding window (without the sample itself)
            mask_center = np.ones(2 * w + 1, dtype=bool)
            mask_center[w] = False
            neighbor_windows = windows[:, mask_center]
            baseline = np.nanmedian(neighbor_windows, axis=1)

            left  = np.empty(n); left[:]  = np.nan
            right = np.empty(n); right[:] = np.nan
            left[1:]  = y[:-1]
            right[:-1] = y[1:]

            mid = y
            finite_mask = np.isfinite(left) & np.isfinite(mid) & np.isfinite(right)
            
            # Check if sample point is a local extreme, greater than a user defined threshold and is
            # greater than the threshold value
            is_local_extreme = ((mid > left) & (mid > right)) | ((mid < left) & (mid < right))
            exceeds_min_diff = (np.abs(mid - left) > self.min_diff_threshold) & \
                               (np.abs(mid - right) > self.min_diff_threshold)
            is_spike = finite_mask & is_local_extreme & exceeds_min_diff & \
                       (np.abs(mid - baseline) > t)
            spike_mask[1:-1] |= is_spike[1:-1]

            if self.apply_correction:
                # Make copy of data to avoid interpolating any existing corrections
                y_corrected = y.copy()
                spike_indices = np.where(spike_mask)[0]
                if spike_indices.size:
                    good_mask = ~spike_mask & np.isfinite(y)
                    good_idx = np.where(good_mask)[0]
                    if good_idx.size:
                        y_corrected[spike_indices] = np.interp(spike_indices, good_idx, y[good_idx])
                    else:
                        y_corrected[spike_indices] = np.nan
                event_data[ch] = y_corrected
            else:
                # Replace sample with NaN if corrections aren't applied
                y[spike_mask] = np.nan

        return event
    
def remove_spikes(data, results, k: float = 5.0, edge_scale: float=2.6, in_place: bool =True, apply_correction=False, min_diff_threshold: float = 0.0, baseline_lookback_window = 1, source_key=None):
    analysis_data = None
    
    if source_key and results.get(source_key) is not None:
        analysis_data = results[source_key] if in_place else results[source_key].copy()
    else:
        analysis_data = data if (in_place and data.flags.writeable) else data.copy()

    for event in analysis_data:
        try:
            event_data = event
            num_channels, n = event_data.shape
            w = max(baseline_lookback_window, 1)

            for ch in range(num_channels):
                y = event_data[ch]

                if n < 2:
                    continue
                
                # Calculate differences between consecutive samples
                diffs = np.abs(np.diff(y))
                diffs = diffs[np.isfinite(diffs)]

                # Get the median difference between samples
                s = float(np.median(diffs)) if diffs.size else 0.0
                s = max(s, np.finfo(float).eps)

                # Threshold values for how much a sample can differ from the median of the window
                t = k * s
                t_edge = edge_scale * t

                spike_mask = np.zeros(n, dtype=bool)

                # edges
                if np.isfinite(y[0]) and np.isfinite(y[1]) and abs(y[0] - y[1]) > t_edge:
                    spike_mask[0] = True
                if np.isfinite(y[-1]) and np.isfinite(y[-2]) and abs(y[-1] - y[-2]) > t_edge:
                    spike_mask[-1] = True

                
                y_nan = np.where(np.isfinite(y), y, np.nan)
                pad = np.pad(y_nan, w, constant_values=np.nan)

                # Split data into sliding window
                shape = (n, 2 * w + 1)
                strides = (pad.strides[0], pad.strides[0])
                windows = np.lib.stride_tricks.as_strided(pad, shape=shape, strides=strides)

                # Calculate median of samples in the sliding window (without the sample itself)
                mask_center = np.ones(2 * w + 1, dtype=bool)
                mask_center[w] = False
                neighbor_windows = windows[:, mask_center]
                baseline = np.nanmedian(neighbor_windows, axis=1)

                left  = np.empty(n); left[:]  = np.nan
                right = np.empty(n); right[:] = np.nan
                left[1:]  = y[:-1]
                right[:-1] = y[1:]

                mid = y
                finite_mask = np.isfinite(left) & np.isfinite(mid) & np.isfinite(right)
                
                # Check if sample point is a local extreme, greater than a user defined threshold and is
                # greater than the threshold value
                is_local_extreme = ((mid > left) & (mid > right)) | ((mid < left) & (mid < right))
                exceeds_min_diff = (np.abs(mid - left) > min_diff_threshold) & \
                                (np.abs(mid - right) > min_diff_threshold)
                is_spike = finite_mask & is_local_extreme & exceeds_min_diff & \
                        (np.abs(mid - baseline) > t)
                spike_mask[1:-1] |= is_spike[1:-1]

                if apply_correction:
                    # Make copy of data to avoid interpolating any existing corrections
                    y_corrected = y.copy()
                    spike_indices = np.where(spike_mask)[0]
                    if spike_indices.size:
                        good_mask = ~spike_mask & np.isfinite(y)
                        good_idx = np.where(good_mask)[0]
                        if good_idx.size:
                            y_corrected[spike_indices] = np.interp(spike_indices, good_idx, y[good_idx])
                        else:
                            y_corrected[spike_indices] = np.nan
                    event_data[ch] = y_corrected
                else:
                    # Replace sample with NaN if corrections aren't applied
                    y[spike_mask] = np.nan

        except (KeyError, IndexError, TypeError) as e:
            print(e)
    results["spike_filtered"] = data
    return data

