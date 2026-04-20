from . import templates as ft

from tqdm import tqdm
import numpy as np

class PruneFilter(ft.Filter):
    """Drop events where the listed channels have mismatched data lengths
    or misaligned first timestamps. Returns None to drop, event to pass."""

    def __init__(self, channels: list[int], *, n_samples=None, in_place=True, check_timing=True):
        super().__init__(in_place=in_place)
        self.channels = channels
        self.n_samples = n_samples
        self.samp_set = {n_samples} if n_samples else set()
        self._ch0 = channels[0]
        self._rest = channels[1:]
        self._check_n = n_samples is not None
        self._check_timing = check_timing

    def _process(self, event) -> dict | None:
        try:
            data = event["data"] if isinstance(event["data"], np.ndarray) else np.asarray(event["data"])
            timing = event["timing"]
            ch0 = self._ch0
            ref_len = len(data[ch0])
            ref_ts = timing[ch0][0] if timing else None
            for ch in self._rest:
                if len(data[ch]) != ref_len:
                    return None
                if self._check_timing and timing[ch][0] != ref_ts:
                    return None
            if self._check_n and ref_len != self.n_samples:
                return None
        except (KeyError, IndexError, TypeError, Exception) as e:
            print(e)
            return None
        return event
        

class MinDataLengthFilter(ft.Filter):
    """Drop events where the listed channels have data shorter than n_samples.
    Returns None to drop, event to pass."""

    def __init__(self, channels: list[int], n_samples, in_place=True):
        super().__init__(in_place=in_place)
        self.channels = channels
        self.n_samples = n_samples
        self.samp_set = {n_samples}
        self._ch0 = channels[0]
        self._rest = channels[1:]

    def _process(self, event) -> dict | None:
        try:
            data = event["data"] if isinstance(event["data"], np.ndarray) else np.asarray(event["data"])
            n = self.n_samples
            if len(data[self._ch0]) != n:
                return None
            for ch in self._rest:
                if len(data[ch]) != n:
                    return None
        except (KeyError, IndexError, TypeError, Exception) as e:
            print(e)
            return None
        return event

def convert_acq_to_npy(acq, indices=[]):
    evt_keys = ['data', 'window_labels', 'time', 'timing', 'ecc_errors']

    corrected_evts = []
    for i in tqdm(range(len(acq)) if len(indices) >= 0 else indices, desc="Converting lists to numpy arrays"):
        try:
            evt = {}
            for evt_key in evt_keys:
                evt[evt_key] = np.asarray(acq[i][evt_key])
            corrected_evts.append(evt)
        except:
            print(f"Could not load event at index {i}")
    return corrected_evts

class RolloverFilter(ft.Filter):
    """Correct rollovers using a minimum threshol and specified correction amount"""

    def __init__(self, *, min_threshold: int, in_place: bool = True, apply_correction=False, correction = 0):
        super().__init__(in_place=in_place)
        self.min_threshold = min_threshold
        self.apply_correction = apply_correction
        self.correction = correction

    def _process(self, event) -> dict | None:
        try:
            data = event["data"] if isinstance(event["data"], np.ndarray) else np.asarray(event["data"])
            if (self.apply_correction):
               data[data < self.min_threshold] += self.correction
            else:
                data[data < self.min_threshold] = np.nan
        except (KeyError, IndexError, TypeError, Exception) as e:
            print(e)
            return None
        return event
    


class SpikesFilter(ft.Filter):
    """Correct rollovers using a minimum threshol and specified correction amount"""

    def __init__(self, k: float = 5.0, edge_scale: float=2.6, in_place: bool =True, apply_correction=False, min_diff_threshold: float = 0.0, baseline_lookback_window = 0):
        super().__init__(in_place=in_place)
        self.k = k
        self.edge_scale = edge_scale
        self.apply_correction = apply_correction
        self.min_diff_threshold = min_diff_threshold
        self.baseline_lookback_window = baseline_lookback_window
    def _process(self, event) -> dict | None:
        try:
            event_data = event["data"] if isinstance(event["data"], np.ndarray) else np.asarray(event["data"])
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

        except (KeyError, IndexError, TypeError):
            return None
        return event
    


