from __future__ import annotations
from IPython.display import display, clear_output
from ipywidgets import widgets
from naluacq import Acquisition
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .save_plots import *


def _make_nav_controls(max_events, n_channels, state, render_fn):
    prev_btn = widgets.Button(description="Prev", layout=widgets.Layout(width="80px"))
    next_btn = widgets.Button(description="Next", layout=widgets.Layout(width="80px"))
    event_box = widgets.BoundedIntText(
        value=0, min=0, max=max_events - 1, step=1, description="Event"
    )
    channel_box = widgets.BoundedIntText(
        value=state["channel"], min=0, max=n_channels - 1, step=1, description="Channel"
    )

    _busy = [False]

    def set_event(new_i):
        if _busy[0]:
            return
        _busy[0] = True
        new_i = int(np.clip(new_i, 0, max_events - 1))
        state["event_index"] = new_i
        event_box.value = new_i
        render_fn()
        _busy[0] = False

    def set_channel(new_ch):
        if _busy[0]:
            return
        _busy[0] = True
        new_ch = int(np.clip(new_ch, 0, n_channels - 1))
        state["channel"] = new_ch
        channel_box.value = new_ch
        render_fn()
        _busy[0] = False

    prev_btn.on_click(lambda _: set_event(state["event_index"] - 1))
    next_btn.on_click(lambda _: set_event(state["event_index"] + 1))
    event_box.observe(lambda c: set_event(c["new"]), names="value")
    channel_box.observe(lambda c: set_channel(c["new"]), names="value")

    return widgets.HBox([prev_btn, next_btn, event_box, channel_box])


def interactive_plot(
    acq: Acquisition | np.ndarray,
    channel: int = 0,
    title: str = "",
    filters=None,
    sharey=False,
    exclude_channels=[],
    trigger_level=0,
    signal_level=0,
    yticks=None,
    save_dir=None,
) -> None:
    if filters is None:
        filters = []

    if isinstance(acq, Acquisition):
        n_channels = int(getattr(acq, "params", {}).get("channels", np.asarray(acq[0]["data"]).shape[0]))
    elif isinstance(acq, np.ndarray):
        n_channels = int(acq.shape[1])
    else:
        raise TypeError(f"acq must be Acquisition or np.ndarray, got {type(acq)}")

    max_events = len(acq)
    state = {"event_index": 0, "channel": int(channel)}

    def fetch_trace(event_index):
        event = acq[event_index]
        if isinstance(acq, Acquisition):
            y = np.array(event["data"], dtype=float)
        else:
            y = np.array(event, dtype=float)
        for ch_idx in range(y.shape[0]):
            trace = y[ch_idx]
            for f in filters:
                trace = f(trace)
            y[ch_idx] = trace
        return y

    # Build the figure once inside an Output widget so it stays in one place
    fig_out = widgets.Output()
    with fig_out:
        fig, axes = plt.subplots(
            n_channels, 1,
            figsize=(10, round(3 * n_channels)),
            sharex=True,
            sharey=sharey,
        )
        axes = np.atleast_1d(axes)
        lines = []
        for ch_idx, ax in enumerate(axes):
            (ln,) = ax.plot([], [], lw=1)
            lines.append(ln)
            ax.set_title(f"Channel {ch_idx}")
            ax.set_ylabel("Counts")
            ax.set_xlabel("Samples [n]")
            ax.grid(True)
            if yticks:
                ax.set_yticks(yticks)
            if trigger_level:
                ax.axhline(trigger_level)
            if signal_level:
                ax.axhline(signal_level, color="red", linestyle="dashed")
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def render():
        i = state["event_index"]
        event_data = fetch_trace(i)
        fig.suptitle(f"{title} - Event {i}")
        for ch_idx in range(n_channels):
            y = np.asarray(event_data[ch_idx])
            lines[ch_idx].set_data(np.arange(y.size), y)
            axes[ch_idx].relim()
            axes[ch_idx].autoscale_view()
        fig.canvas.draw_idle()

    controls = _make_nav_controls(max_events, n_channels, state, render)
    render()
    display(widgets.VBox([controls, fig_out]))


def interactive_plot_psd(
    acq_0: Acquisition | np.ndarray,
    acq_1: Acquisition | np.ndarray,
    channel: int = 0,
    title: str = "",
    filters=None,
) -> None:
    if filters is None:
        filters = []

    if isinstance(acq_0, Acquisition):
        n_channels = int(getattr(acq_0, "params", {}).get("channels", np.asarray(acq_0[0]["data"]).shape[0]))
    elif isinstance(acq_0, np.ndarray):
        n_channels = int(acq_0.shape[1])
    else:
        raise TypeError(f"acq_0 must be Acquisition or np.ndarray, got {type(acq_0)}")

    max_events = len(acq_0)
    state = {"event_index": 0, "channel": int(channel)}

    def fetch_trace(acq, event_index):
        event = acq[event_index]
        if isinstance(acq, Acquisition):
            y = np.array(event["data"], dtype=float)
        else:
            y = np.array(event, dtype=float)
        for ch_idx in range(y.shape[0]):
            trace = y[ch_idx]
            for f in filters:
                trace = f(trace)
            y[ch_idx] = trace
        return y

    fig_out = widgets.Output()
    with fig_out:
        fig, axes = plt.subplots(n_channels, 2, figsize=(10, round(1.5 * n_channels)))
        axes = np.atleast_2d(axes)
        lines = [[None, None] for _ in range(n_channels)]
        for ch_idx in range(n_channels):
            (ln0,) = axes[ch_idx, 0].plot([], [], lw=1)
            (ln1,) = axes[ch_idx, 1].plot([], [], lw=1)
            lines[ch_idx][0] = ln0
            lines[ch_idx][1] = ln1
            axes[ch_idx, 0].set_title(f"Channel {ch_idx}")
            axes[ch_idx, 1].set_title(f"Channel {ch_idx}")
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def render():
        i = state["event_index"]
        event_data_0 = fetch_trace(acq_0, i)
        event_data_1 = fetch_trace(acq_1, i)
        fig.suptitle(f"{title} - Event {i}")
        for ch_idx in range(n_channels):
            y0 = np.asarray(event_data_0[ch_idx])
            lines[ch_idx][0].set_data(np.arange(y0.size), y0)
            axes[ch_idx, 0].relim()
            axes[ch_idx, 0].autoscale_view()

            y1 = np.asarray(event_data_1[ch_idx])
            lines[ch_idx][1].set_data(np.arange(y1.size), y1)
            axes[ch_idx, 1].relim()
            axes[ch_idx, 1].autoscale_view()
        fig.canvas.draw_idle()

    controls = _make_nav_controls(max_events, n_channels, state, render)
    render()
    display(widgets.VBox([controls, fig_out]))
