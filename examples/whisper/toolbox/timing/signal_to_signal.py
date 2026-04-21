import itertools
import numpy as np
from typing import Optional, Tuple

from .trigger_to_signal import find_events_of_interest


def find_channels_with_events(
    data: np.ndarray,
    results: dict,
    event_locs: Optional[np.ndarray] = None,
    channels_per_event: Optional[dict] = None,
) -> list[int]:
    """Filters for events that have an event of interest occurring in addition to the trigger edge. Keeps track of the number of times a channel has an event of interest.

    Args:
        event_locs (np.ndarray, optional): Shape (N, 2) event/channel pairs.
        Defaults to None (auto-loaded from results["trig2sig_filtered_events"]).
        channels_per_event (dict, optional): Maps events to sets of channels with crossings.
        Defaults to None (auto-loaded from results["trig2sig_channels_per_event"]).

    Returns:
        list[int]: A list of channel indices.
    """
    if event_locs is None:
        event_locs = results["trig2sig_filtered_events"]
    if channels_per_event is None:
        channels_per_event = results["trig2sig_channels_per_event"]

    unique_events = np.unique(event_locs[:, 0])

    channels_with_multiple_events = []
    for unique_event in unique_events:
        if len(channels_per_event[unique_event]) >= 2:
            for channel in channels_per_event[unique_event]:
                channels_with_multiple_events.append(channel)

    results["sig2sig_channels_with_multiple_events"] = channels_with_multiple_events
    return channels_with_multiple_events


def find_channel_pairings(
    data: np.ndarray,
    results: dict,
    event_locs: Optional[np.ndarray] = None,
    channels_per_event: Optional[dict] = None,
) -> dict:
    """Determine the number of times a channel pairing occurs. When event_locs or channels_per_event are None, they are loaded from results automatically.

    Args:
        event_locs (np.ndarray, optional): Shape (N, 2) event/channel pairs.
        Defaults to None (auto-loaded from results["trig2sig_filtered_events"]).
        channels_per_event (dict, optional): Maps events to sets of channels with crossings.
        Defaults to None (auto-loaded from results["trig2sig_channels_per_event"]).

    Returns:
        dict: Keys are channel pairings (e.g. "0-1"), values are occurrence counts.
    """
    if event_locs is None:
        event_locs = results["trig2sig_filtered_events"]
    if channels_per_event is None:
        channels_per_event = results["trig2sig_channels_per_event"]

    signal_locs = event_locs[event_locs[:, 1] != 7]
    unique_events = np.unique(signal_locs[:, 0])
    possible_channel_pairings = list(itertools.combinations(np.arange(0, 8), 2))

    occurences = []
    event_channel_pairs = []

    for unique_event in unique_events:
        if len(channels_per_event[unique_event]) >= 2:
            event_channel_pairs.append(
                list(itertools.combinations(channels_per_event[unique_event], 2))
            )

    for channel_pairs in event_channel_pairs:
        for channel_pair in channel_pairs:
            channel0, channel1 = channel_pair
            for combo_idx, channel_pairing in enumerate(possible_channel_pairings):
                channel_0, channel_1 = channel_pairing
                if (channel0 == channel_0 and channel1 == channel_1) or (
                    channel1 == channel_0 and channel0 == channel_1
                ):
                    occurences.append(combo_idx)
                    break

    unique_idx, counts = np.unique(occurences, return_counts=True)

    counter = {}
    for idx, count in zip(unique_idx, counts):
        channel_a, channel_b = possible_channel_pairings[idx]
        if channel_a == 7 or channel_b == 7:
            continue
        counter[f"{channel_a}-{channel_b}"] = count

    results["sig2sig_channel_pairs"] = np.array([key for key in counter.keys()])
    results["sig2sig_channel_pair_counts"] = np.array([value for value in counter.values()])

    results["sig2sig_channel_pairings"] = counter
    return counter


def signal_to_signal_timings(
    data: np.ndarray,
    results: dict,
    thresholds_x: Optional[np.ndarray] = None,
    filter_sig_before_trig = True
) -> dict:
    """Calculate the difference (in samples) between signals of interest in different channels within the same event.

    Args:
        thresholds_x (np.ndarray, optional): Shape (events, channels) crossing indices.
        Defaults to None (auto-loaded from results["threshold_x"]).

    Returns:
        dict: Keys are channel pairings (e.g. "0-1"), values are lists of timing differences.
    """
    if thresholds_x is None:
        thresholds_x = results["threshold_x"]

    if "trig2sig_filtered_events" in results and "trig2sig_channels_per_event" in results:
        filtered_events    = results["trig2sig_filtered_events"]
        channels_per_event = results["trig2sig_channels_per_event"]

        if filter_sig_before_trig and "trig2sig_delta_n" in results and "trig2sig_event_indices" in results:
            delta_n      = results["trig2sig_delta_n"]
            event_idx    = results["trig2sig_event_indices"]
            bad_events   = set(np.asarray(event_idx)[np.asarray(delta_n) <= 0].tolist())
            keep_mask    = np.array([int(e) not in bad_events for e in filtered_events[:, 0]])
            filtered_events    = filtered_events[keep_mask]
            channels_per_event = {k: v for k, v in channels_per_event.items()
                                  if int(k) not in bad_events}
    else:
        filtered_events, channels_per_event = find_events_of_interest(data, results, thresholds_x)

    # Event/channel pairs excluding trigger channel

    signal_locs = filtered_events[filtered_events[:, 1] != 7]
    event_indices = signal_locs[:, 0]
    unique_events = np.unique(event_indices)

    possible_channel_pairings = list(itertools.combinations(np.arange(0, 8), 2))

    timings = {}
    timing_events = {}
    for channel_a, channel_b in possible_channel_pairings:
        if channel_a != 7 and channel_b != 7:
            timings[f"{channel_a}-{channel_b}"] = []
            timing_events[f"{channel_a}-{channel_b}"] = []


    for unique_event in unique_events:
        if len(channels_per_event[unique_event]) >= 2:
            for combo in itertools.combinations(channels_per_event[unique_event], 2):
                channel_a, channel_b = combo
                if channel_a != 7 and channel_b != 7:
                    timings[f"{channel_a}-{channel_b}"].append(
                        thresholds_x[unique_event][channel_a]
                        - thresholds_x[unique_event][channel_b]
                    )
                    timing_events[f"{channel_a}-{channel_b}"].append(
                        unique_event
                    )
    results["sig2sig_channel_pairs"] = []
    for channel_pair, timing in timings.items():
        results["sig2sig_channel_pairs"].append(channel_pair)
        results[f"sig2sig-{channel_pair}-timings"] = timing
        results[f"sig2sig-{channel_pair}-source_events"] = timing_events
    results["sig2sig_timings"] = timings
    results["sig2sig_source_events"] = timing_events
    return timings
