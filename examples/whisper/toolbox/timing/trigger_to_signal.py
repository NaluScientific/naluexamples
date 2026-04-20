
from collections import defaultdict
import numpy as np
from typing import Optional, Tuple


def find_events_of_interest(
    data: np.ndarray,
    results: dict,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    
    if thresholds is None:
        thresholds = results["threshold_x"]

    # Create an array of (Event, Channel) pairs where an event occurs
    event_locs = np.argwhere(~np.isnan(thresholds))

    # Map each event to its set of channels with crossings
    channels_per_event = defaultdict(set)
    for event, channel in event_locs:
        channels_per_event[event].add(channel)

    # Filter out events that contain only the trigger channel (7)
    trigger_only_events = {
        event for event, channels in channels_per_event.items() if channels == {7}
    }

    filtered_events = np.array(
        [
            [event, channel]
            for event, channel in event_locs
            if event not in trigger_only_events
        ]
    )
    results["trig2sig_filtered_events"] = filtered_events
    results["trig2sig_channels_per_event"] = channels_per_event
    return filtered_events, channels_per_event


def trigger_to_signal_timings(
    data: np.ndarray,
    results: dict,
    event_locs: Optional[np.ndarray] = None,
    thresholds_x: Optional[np.ndarray] = None,
    filter_signals_before_trig: bool = False,
    trig_channel: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if event_locs is None:
        event_locs = results["trig2sig_filtered_events"]
    if thresholds_x is None:
        thresholds_x = results["threshold_x"]

    # Event/channel pairs excluding the trigger channel
    signal_locs = event_locs[event_locs[:, 1] != trig_channel]
    event_indices, channel_indices = signal_locs[:, 0], signal_locs[:, 1]

    signal_thresholds_x = thresholds_x[event_indices, channel_indices]
    trigger_thresholds_x = thresholds_x[event_indices, trig_channel]

    delta_n = signal_thresholds_x - trigger_thresholds_x

    if filter_signals_before_trig:
        mask = np.argwhere(delta_n > 0).flatten()
        delta_n = delta_n[mask]
        event_indices = event_indices[mask]
        channel_indices = channel_indices[mask]

    results["trig2sig_delta_n"] = delta_n
    results["trig2sig_event_indices"] = event_indices
    results["trig2sig_channel_indices"] = channel_indices
    return delta_n, event_indices, channel_indices

def filter_signals_before_trig(
        data: np.ndarray,
        results: dict,
        source_key: str = "trig2sig_event_indices",
        save_key: str = "trig2sig_events_after_trig"
):
    results[save_key] = data[results[source_key]]