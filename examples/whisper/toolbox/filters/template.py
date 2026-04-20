from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from typing import Iterator, Dict

Event = Dict

class Filter(ABC):
    """Base class for data filters

    Variable number of arbuments

    Args:
        in_place: overwrite existing or
    """
    
    def __init__(self, *, in_place: bool = True):
        self.in_place = in_place
    
    def run(self, events: Iterator[dict]) -> Iterator[dict]:
        for event in events:
            if not self.in_place:
                event = deepcopy(event)
            result = self._process(event)
            if result is not None:  # allows filters to drop events
                yield result
    
    @abstractmethod
    def _process(self, event: Event) -> Event | None:
        """Process a single event. Return None to drop it."""
        pass


class FilterPipeline():
    """Chain multiple filters together."""
    
    def __init__(self, filters = None, in_place: bool = True):
        #super().__init__(in_place = in_place)
        self.filters = filters or []
    
    def add(self, f: Filter):
        """Return itself so it can be chained using builder pattern
        filt.add(a).add(b).add(c) etc
        """
        self.filters.append(f)
        return self
    
    def run(self, events: Iterator[dict]) -> Iterator[dict]:
        """Run all the filters on all the events

        We should probably make this one multiprocess using tqdm or soemthing.
        """
        for f in self.filters:
            events = f.run(events)
        yield from events

    

