"""Reader interface. Section and plotting code depends only on this."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ep_seismic.volume.domain import SampleDomain
from ep_seismic.volume.index import GeometryIndex

DEFAULT_MAX_BYTES = 2 * 1024**3


class SeismicSource(Protocol):
    """A seismic file read in place, trace by trace.

    Implementations must not load the whole file. ``read_traces`` must refuse
    requests larger than ``max_bytes`` instead of allocating them.
    """

    n_traces: int
    n_samples: int

    @property
    def samples(self) -> np.ndarray:
        """Sample axis: ms if time, m or ft if depth (the domain says which)."""
        ...

    @property
    def domain(self) -> SampleDomain | None:
        """Vertical domain and quantity, or None if not yet settled."""
        ...

    def require_domain(self) -> SampleDomain:
        """The domain, or raise ``DomainUnspecified`` saying what is missing."""
        ...

    def read_traces(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        sample_slice: slice | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> np.ndarray:
        """Return ``float32`` samples, shape ``(len(indices), n_out_samples)``."""
        ...

    def index(
        self, *, cache_dir: str | None = None, rebuild: bool = False
    ) -> GeometryIndex:
        """Load the sidecar index if it is current, else build and save it."""
        ...

    def close(self) -> None: ...
