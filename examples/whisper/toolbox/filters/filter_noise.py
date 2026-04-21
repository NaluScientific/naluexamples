from .template import Filter, FilterPipeline

import numpy as np
import scipy
from typing import Dict

Event = Dict
    
class FilterNoise(Filter):
    """
    Filter signals with a combination of Notch filters or a lowpass butterworth filter.
    """
    def __init__(self, f_s: int, filter_type: str, order=3, q_factor: int =10, freqs: list[int]=[], in_place=True):
        """Creates a filter for noise

        Arguments:
            f_s -- The sampling frequency in Hz
            filter_type -- `notch` or `butter`

        Keyword Arguments:
            order -- The order of the Butterworth filter (default: {3})
            q_factor -- The quality factor of the notch filter (default: {10})
            freqs --  f_s (int): The sampling frequency in Hz.
            filter_type (str): The list of frequencies to be filtered. Only uses the
            first frequency in the list if filter_type is `butterworth`. 
            in_place -- Determines if corrections are applied in place.
            Defaults to True.  (default: {True})
        """
        super().__init__(in_place=in_place)
        self.f_s = f_s
        self.filter_type = filter_type
        self.order = order
        self.q_factor = q_factor
        self.freqs = freqs
    
    def filter_cw_noise(self, signal: np.ndarray, filter_type: str, f_s: int, order:int = 3, q_factor: int = 10, freqs : list[int]=[]) -> np.ndarray:
        """Filter CW noise

        Arguments:
            signal -- Data from a single channel
            filter_type -- `notch` or `butter`
            f_s -- The sampling frequency in Hz

        Keyword Arguments:
            order -- The order of the BUtterworth filter (default: {3})
            q_factor -- The quality factor of the notch filter (default: {10})
            freqs -- The lit of frequencies to be filtered. Only uses the first frequency in the list if filter_type is `butterworth` (default: {[]})

        Returns:
            Filtered channel data
        """
        filtered_signal = signal
        
        for freq in freqs:
            if filter_type == "notch":
                b, a = scipy.signal.iirnotch(freq, q_factor, f_s)
                filtered_signal = scipy.signal.filtfilt(b, a, filtered_signal)
            else:
                b, a = scipy.signal.butter(order, freq, fs=f_s)
                filtered_signal = scipy.signal.filtfilt(b, a, filtered_signal)
                break
                
        return filtered_signal


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

        event["data"] = np.apply_along_axis(self.filter_cw_noise, -1, event_data, self.filter_type, self.f_s, self.order, self.q_factor, self.freqs)

        return event
    

def filter_noise(
    data,
    results: dict,
    source_key = None,
    save_key = "noise_filtered",
    f_s: int = 1,
    filter_type: str = "notch",
    order: int = 3,
    q_factor: int = 10,
    freqs: list[int] = [],
    in_place=True
):
    analysis_data = None
    
    if source_key and results.get(source_key) is not None:
        analysis_data = results[source_key] if in_place else results[source_key].copy()
    else:
        analysis_data = data if (in_place and data.flags.writeable) else data.copy()

    for event in analysis_data:
        try:
            if filter_type == "notch":
                for freq in freqs:
                    b, a = scipy.signal.iirnotch(freq, q_factor, f_s)
                    event[:] = scipy.signal.filtfilt(b, a, event, axis=-1)
            elif filter_type == "bandpass":
                if len(freqs) < 2:
                    raise ValueError("bandpass requires freqs=[low, high]")
                b, a = scipy.signal.butter(order, freqs[:2], btype="band", fs=f_s)
                event[:] = scipy.signal.filtfilt(b, a, event, axis=-1)
            else:  # lowpass / butter
                b, a = scipy.signal.butter(order, freqs[0], btype="low", fs=f_s)
                event[:] = scipy.signal.filtfilt(b, a, event, axis=-1)

        except (KeyError, IndexError, TypeError, Exception) as e:
            print(e)

    results[save_key] = analysis_data
    return analysis_data


def psd(data, results, f_s, in_place=True):
    noise_data = results["noise_data"]
    signal_data = results["signal_data"]
    trig_noise_data = results["trig_noise_data"]
    trig_signal_data = results["trig_signal_data"]
    noise_freq, noise_p = scipy.signal.welch(noise_data, f_s)
    signal_freq, signal_p = scipy.signal.welch(signal_data, f_s)
    trig_noise_freq, trig_noise_p = scipy.signal.welch(trig_noise_data, f_s)
    trig_signal_freq, trig_signal_p = scipy.signal.welch(trig_signal_data, f_s)
    results["noise_freq"] = noise_freq
    results["noise_p"] = noise_p
    results["signal_freq"] = signal_freq
    results["signal_p"] = signal_p
    results["trig_noise_freq"] = trig_noise_freq
    results["trig_noise_p"] = trig_noise_p
    results["trig_signal_freq"] = trig_signal_freq
    results["trig_signal_p"] = trig_signal_p
