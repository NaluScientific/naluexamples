from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
import numpy as np
from copy import deepcopy
from typing import Any, TypeAlias, TYPE_CHECKING
from tqdm import tqdm


Event: TypeAlias = dict[str, Any]

class Filter(ABC):
    """Base class for data filters

    Variable number of arbuments

    Args:
        in_place: overwrite existing or
    """
    
    def __init__(self, *, in_place: bool = True):
        self.in_place = in_place
    
    def run(self, events: Iterable[dict]) -> Iterable[dict]:
        _process = self._process
        if not self.in_place:
            event = deepcopy(event)
        for event in events:
            result = _process(event)
            if result is not None:
                yield result

    def process_one(self, event: Event) -> Event | None:
        """Process a single event without generator overhead."""
        if not self.in_place:
            event = deepcopy(event)
        return self._process(event)

    @abstractmethod
    def _process(self, event: Event) -> Event | None:
        """Process a single event. Return None to drop it."""
        raise NotImplementedError


class FilterPipeline:
    """Chain multiple filters together."""
    
    def __init__(self, filters = None, in_place: bool = True):
        self.in_place = in_place
        self.filters = filters if filters else []
    
    def add(self, f: Filter):
        """Return itself so it can be chained using builder pattern
        filt.add(a).add(b).add(c) etc
        """
        self.filters.append(f)
        return self
    
    def run(self, events: Iterator[Event]) -> Iterator[Event]:
        """Run all the filters on all the events

        We should probably make this one multiprocess using tqdm or soemthing.
        """
        for f in self.filters:
            events = f.run(events=events)
        yield from events

    def process_one(self, event: Event) -> Event | None:
        """Process a single event through all filters without generator overhead.

        Short-circuits on the first filter that returns None.
        """
        for f in self.filters:
            event = f.process_one(event)
            if event is None:
                return None
        return event

    def __call__(self, events: Iterator[Event]) -> Iterator[Event]:
        return self.run(events)

    def collect(self, events: Event) -> list[Event]:
        return list(self.run(events))

    def collect_results(self, events: Iterator[Event], exclude_keys: set[str] | None = None) -> "Results":
        """Run the pipeline and collect only analysis outputs.

        Returns a Results object containing small per-event values
        (scalars, 1D arrays) while letting the large waveform data
        be garbage collected.
        """
        from .results import Results
        results = Results(exclude_keys=exclude_keys)
        for event in self.run(events):
            results.append(event)
        return results