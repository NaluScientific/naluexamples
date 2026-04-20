from toolbox.baseline.baseline import *
from toolbox.filters.events import *
from toolbox.filters.filter_noise import *
from toolbox.filters.filter_rollover import *
from toolbox.filters.filter_spikes import *
from toolbox.timing import *
from .analysis_pipeline import *

rollover_step = Step(
    name="Filter Rollover",
    fn=remove_rollover,
    requires=frozenset(),
    produces=frozenset({"rollover_corrected"}),
)

spike_step = Step(
    name="Filter Digitizer Spikes",
    fn=remove_spikes,
    requires=frozenset(),
    produces=frozenset({"spike_filtered"}),
)

noise_step = Step(
    name="Noise Filtered",
    fn=filter_noise,
    requires=frozenset(),
    produces=frozenset({"noise_filtered"}),
)

baseline_step = Step(
    name="Baseline",
    fn=find_baseline_samples,
    requires=frozenset(),
    produces=frozenset({
        "baselines_sliding_window",
        "baseline_sliding_window_events",
        "baseline_sliding_window_start_idx",
        "baselines_fixed_window",
        "baseline_fixed_window_events",
        "baseline_fixed_window_win_idx",
        "baselines_0_window",
    }),
)

signal_noise_step = Step(
    name="Signal vs Noise",
    fn=separate_signal_and_noise,
    requires=frozenset(),
    produces=frozenset({
        "signal_events", "signal_channels",
        "noise_events", "noise_channels",
        "trig_signal_events", "trig_signal_channels",
        "trig_noise_events", "trig_noise_channels",
        "noise_data", "signal_data", "trig_signal_data", "trig_noise_data",
    }),
    index_keys=frozenset({
        "signal_events", "noise_events",
        "trig_signal_events", "trig_noise_events",
    }),
)

psd_step = Step(
    name="PSD",
    fn=psd,
    requires=frozenset({"noise_data", "signal_data", "trig_noise_data", "trig_signal_data"}),
    produces=frozenset({
        "noise_freq", "noise_p",
        "signal_freq", "signal_p",
        "trig_noise_freq", "trig_noise_p",
        "trig_signal_freq", "trig_signal_p",
    }),
)

# --- Timing steps ---

peaks_step = Step(
    name="Find Peaks",
    fn=find_peaks,
    requires=frozenset(),
    produces=frozenset({"channel_min_peaks", "channel_max_peaks"}),
)

thresholds_step = Step(
    name="Calculate Thresholds",
    fn=calculate_thresholds,
    requires=frozenset(),
    produces=frozenset({"thresholds"}),
)

threshold_x_step = Step(
    name="Find Threshold Crossings",
    fn=find_threshold_x,
    requires=frozenset({"thresholds"}),
    produces=frozenset({"threshold_x"}),
)

events_of_interest_step = Step(
    name="Find Events of Interest",
    fn=find_events_of_interest,
    requires=frozenset({"threshold_x"}),
    produces=frozenset({"trig2sig_filtered_events", "trig2sig_channels_per_event"}),
    index_keys=frozenset({"trig2sig_filtered_events"}),
)

trig_to_signal_step = Step(
    name="Trigger to Signal Timings",
    fn=trigger_to_signal_timings,
    requires=frozenset({"trig2sig_filtered_events", "threshold_x"}),
    produces=frozenset({"trig2sig_delta_n", "trig2sig_event_indices", "trig2sig_channel_indices"}),
    index_keys=frozenset({"trig2sig_event_indices"}),
)

channels_with_events_step = Step(
    name="Channels with Events",
    fn=find_channels_with_events,
    requires=frozenset({"trig2sig_filtered_events", "trig2sig_channels_per_event"}),
    produces=frozenset({"sig2sig_channels_with_multiple_events"}),
)

channel_pairings_step = Step(
    name="Channel Pairings",
    fn=find_channel_pairings,
    requires=frozenset({"trig2sig_filtered_events", "trig2sig_channels_per_event"}),
    produces=frozenset({
        "sig2sig_channel_pairs",
        "sig2sig_channel_pair_counts",
        "sig2sig_channel_pairings",
    }),
)

signal_to_signal_step = Step(
    name="Signal to Signal Timings",
    fn=signal_to_signal_timings,
    requires=frozenset({"threshold_x"}),
    produces=frozenset({"sig2sig_timings", "sig2sig_channel_pairs", "sig2sig_source_events"}),
)

signals_before_trig_step = Step(
    name="Filter Signals Before Trigger",
    fn=filter_signals_before_trig,
    requires=frozenset({"trig2sig_event_indices"}),
    produces=frozenset({"trig2sig_events_after_trig"}),
)
