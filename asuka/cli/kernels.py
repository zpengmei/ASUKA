from __future__ import annotations

import argparse

from asuka.kernels import print_kernel_report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Print a report of available ASUKA native CUDA/C++ kernels.")
    p.add_argument("--full", action="store_true", help="Print per-category symbol details.")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    args = p.parse_args(argv)

    print_kernel_report(full=bool(args.full), json_output=bool(args.json))


if __name__ == "__main__":  # pragma: no cover
    main()

