"""Fail when a Python block exceeds the configured complexity limit."""

from __future__ import annotations

import argparse
from pathlib import Path

from radon.complexity import cc_visit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--max", type=int, default=10, dest="maximum")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in sorted(args.path.rglob("*.py")):
        for block in cc_visit(path.read_text(encoding="utf-8")):
            if block.complexity > args.maximum:
                failures.append(
                    f"{path}:{block.lineno} {block.name} "
                    f"has complexity {block.complexity} (maximum {args.maximum})"
                )
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
