from . import templates as ft
from .data_forming_filters import PruneFilter
from .array_builder import ArrayBuilder
import numpy as np
from tqdm import tqdm


class FETPipeline:
    """Filter → Extract → Transform pipeline for building a dataset dict.

    Stage 0 (Filter):    Validation pipeline. PruneFilter is added
                         automatically when channels are known. Runs as a
                         single streaming pass over the full acquisition.
    Stages 1+ (Extract): Optional user-defined filters for per-event
                         selection. Run in sorted stage order only on
                         events that passed stage 0.
    Transform:           ArrayBuilder writes directly into preallocated
                         (N, C, S) arrays; events are not accumulated.

    Convenience shorthands:
        add_filter(f)    — shorthand for add(0, f); stage 0 is always registered.
        add_extractor(f) — shorthand for add(1, f); stage 1 must be registered
                           via add_stage first.

    Lower-level API:
        add_stage(stage, pipeline) — register a pipeline at a given stage number
                                     (required before calling add() for that stage).
        add(stage, f)              — add filter f to a registered stage.

    Args:
        channels:  ASIC channel indices to extract and validate. Pass
                   None to auto-detect non-empty channels from the first
                   passing event (PruneFilter is then not added automatically).
        pedestals: Optional pedestal array forwarded to ArrayBuilder.
    """

    def __init__(self, channels: list[int] | None = None,
                 pedestals: np.ndarray | None = None,
                 check_timing: bool = True):
        self.channels = channels
        self.pedestals = pedestals if isinstance(pedestals, np.ndarray) else np.asarray(pedestals)
        self._check_timing = check_timing
        self._stage_names: dict[int, str|None] = {}
        self._stages: dict[int, ft.FilterPipeline] = {}

        # default stage will prune malformed events.
        _stage0 = ft.FilterPipeline()
        if channels is not None:
            _stage0.add(PruneFilter(channels, check_timing = False))
        self._stages[0] = _stage0

    def add_stage(self, index: int, pipeline: ft.FilterPipeline, name: str|None = None) -> "FETPipeline":
        """Register a pipeline at the given stage number.

        Must be called before add(stage, f) for any stage other than 0 (which is
        registered automatically at init). Stages execute in numerically sorted
        order: stage 0 runs as the streaming filter pass over the full acquisition;
        stages 1, 2, ... run in order during the per-event extract pass on the
        events that passed stage 0.

        Args:
            stage:    Integer stage number (must not already be registered).
            pipeline: Pipeline instance to attach at this stage.

        Raises:
            ValueError: If stage is already registered.
        """
        if index in self._stages.keys():
            raise ValueError(f"Stage {index} is already registered.")
        self._stages[index] = pipeline
        self._stage_names[index] = name
        return self

    def add_filter(self, index: int, f: ft.Filter, name: str|None = None) -> "FETPipeline":
        """Add filter f to the pipeline at the given stage. Returns self for chaining.

        Args:
            stage: Integer stage number. The stage must be registered via add_stage()
                   first (stage 0 is registered automatically at init).
            f:     Any Filter subclass instance.

        Raises:
            ValueError: If stage has not been registered.
        """
        if index not in self._stages.keys():
            raise ValueError(
                f"Stage {index} is not registered. Call add_stage({index}, pipeline) before add()."
            )
        self._stages[index].add(f)
        return self

    def run(self, acquisition) -> dict:
        good_indices, max_samples = self._filter_stage(acquisition)
        print(f"[DEBUG] good_indices count: {len(good_indices)}, max_samples: {max_samples}")
        builder = ArrayBuilder(len(good_indices), self.channels, max_samples, self.pedestals)
        extract_stages = sorted(k for k in self._stages if k > 0)
        has_extractors = bool(extract_stages)
        for idx in tqdm(good_indices, desc="Stage 2+3: extract+transform"):
            event = acquisition[int(idx)]
            if has_extractors:
                for k in extract_stages:
                    event = next(self._stages[k].run([event]), None)
                    if event is None:
                        break
                if event is None:
                    continue
            builder._process(event)
        return builder.to_dict(good_indices)

    def _filter_stage(self, acquisition) -> tuple[list[int], int]:
        """Stage 0: stream all events through the validation pipeline.

        If channels were not provided at init, the active channels are
        detected from the first passing event and stored on self.channels.
        Returns (good_indices, max_samples).
        """
        good_indices = []
        max_samples = 0
        stage0_check = self._stages[0].process_one
        for i in tqdm(range(len(acquisition)), desc="Stage 1: filtering"):
            try:
                event = acquisition[i]
                if stage0_check(event) is not None:
                    if self.channels is None:
                        self.channels = [
                            ch for ch, d in enumerate(event["data"]) if len(d) > 0
                        ]
                    good_indices.append(i)
                    max_samples = max(max_samples, len(event["data"][self.channels[0]]))
            except:
                pass
        return good_indices, max_samples

    def __getitem__(self, index):
        """Get a pipeline stage"""
        if index not in self._stages:
            raise IndexError(
                f"Stage {index} is not registered."
            )
        return self._stage[index]

    def __setitem__(self, key, value):
        """Add a pipeline at index
        
        """
        if key in self._stages:
            raise ValueError(
                f"Stage {key} is already registered. Remove key first"
            )
        self._stages[key] = value

    def __delitem__(self, index):
        """Removes an entire pipeline stage
        """
        if index not in self._stages:
            raise IndexError(
                f"Stage {index} is not registered."
            )
        del self._stages[index]
        
