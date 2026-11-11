"""Plausibility check: is a SEG-Y's vertical axis two-way time or PSDM depth?

    uv run python scripts/check_domain.py FILE.segy [--water-depth 80]
        [--profile ncs_shelf] [--state time|depth] [--traces 300]

Reads a few hundred traces, never the volume, and never sets the domain. Give the
water depth at the survey (from a chart) to make the seabed rule decisive; without it,
the Norwegian-shelf profile is only a weak prior and often cannot separate the two (see
``ep_seismic.volume.domain_check``). ``--state`` compares the result with what you say it is.
"""

import argparse
from pathlib import Path

from ep_seismic.volume import make_domain, open_segy
from ep_seismic.volume.domain_check import WATER_DEPTH_PROFILES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--water-depth", type=float, help="water depth at the survey, m")
    ap.add_argument(
        "--profile", default="ncs_shelf", choices=sorted(WATER_DEPTH_PROFILES)
    )
    ap.add_argument("--traces", type=int, default=300)
    ap.add_argument("--state", choices=("time", "depth"), help="what you say it is")
    ap.add_argument("--depth-unit", choices=("m", "ft"), default="m")
    args = ap.parse_args()
    with open_segy(args.file) as src:
        stated = src.domain
        check = src.check_domain(
            n_traces=args.traces,
            profile=args.profile,
            water_depth=args.water_depth,
            depth_unit=args.depth_unit,
        )
        print(check.report())
        if stated is not None:
            print(f"The file's own name or header already says: {stated.describe()}")
            print("  check against it:", check.against(stated))
        else:
            for hint in src.detect().hints:
                print(f"Hint only: {hint}")
        if args.state:
            domain = make_domain(
                args.state,
                "amplitude",
                vertical_unit="ms" if args.state == "time" else args.depth_unit,
            )
            print(f"Against your statement ({args.state}):", check.against(domain))


if __name__ == "__main__":
    main()
