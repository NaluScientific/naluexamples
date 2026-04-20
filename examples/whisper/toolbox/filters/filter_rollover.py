from .template import Filter, FilterPipeline

import numpy as np
import scipy
from typing import Dict

Event = Dict

class FilterRollover(Filter):
    """
    Masks out any samples that are below a threshold with NaN or add 4096 as a correction
    """
    def __init__(self, min_threshold: int, in_place: bool = True, apply_correction=False):
        """Creates a filter for rollover

        Arguments:
            min_threshold -- The minimum value of a sample that is not considered to be a rollover

        Keyword Arguments:
            in_place -- Determines if corrections are applied in place (default: {True})
            apply_correction -- Determines if 4096 is added to a rollover sample or it is masked out with NaN (default: {False})
        """
        super().__init__(in_place = in_place)
        self.min_threshold = min_threshold
        self.apply_correction = apply_correction

    def _process(self, event: Event) -> Event | None:
        """Processes an event with the filter

        Arguments:
            event -- An event

        Returns:
            An event with the processed data
        """
        if "data" in event:
            event_data = event["data"]
            if not isinstance(event_data, np.ndarray):
                event["data"] = np.array(event_data, dtype="float")
                event_data = event["data"]
            if (self.apply_correction):
                event_data[event_data < self.min_threshold] += 4096
            else:
                event_data[event_data < self.min_threshold] = np.nan
        return event



def remove_rollover(data, results, min_threshold, in_place = True, apply_correction = False, correction = 0):
    try:
        analysis_data = data if (in_place and data.flags.writeable) else data.copy()
        if apply_correction:
            analysis_data[analysis_data < min_threshold] += correction
            results["rollover_corrected"] = analysis_data
        else:
            analysis_data[analysis_data < min_threshold] = np.nan
    except:
        results["rollover_corrected"] = None
    return analysis_data