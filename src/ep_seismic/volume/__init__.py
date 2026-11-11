"""Read-in-place access to very large seismic files, and small extracts from them."""

from ep_seismic.volume.domain import (
    Detection,
    DomainUnspecified,
    SampleDomain,
    detect_domain,
    make_domain,
)
from ep_seismic.volume.domain_check import (
    DomainCheck,
    ambiguity_range,
    check_axis,
    check_domain,
)
from ep_seismic.volume.index import (
    GeometryIndex,
    IndexMeta,
    LatticeIndex,
    TraceIndex,
    cache_dir,
    index_path,
    load_index,
)
from ep_seismic.volume.plot import plot_section
from ep_seismic.volume.section import (
    Section,
    section_crossline,
    section_inline,
    section_path,
)
from ep_seismic.volume.segy import HeaderMap, SegySource, open_segy
from ep_seismic.volume.source import SeismicSource

__version__ = "0.1.0"

__all__ = [
    "Detection",
    "DomainCheck",
    "DomainUnspecified",
    "GeometryIndex",
    "HeaderMap",
    "IndexMeta",
    "LatticeIndex",
    "SampleDomain",
    "Section",
    "SegySource",
    "SeismicSource",
    "TraceIndex",
    "__version__",
    "ambiguity_range",
    "cache_dir",
    "check_axis",
    "check_domain",
    "detect_domain",
    "index_path",
    "load_index",
    "make_domain",
    "open_segy",
    "plot_section",
    "section_crossline",
    "section_inline",
    "section_path",
]
