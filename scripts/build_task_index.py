"""Generate the persistent discovery index for bundled lm-eval tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from lm_eval.tasks._index import INDEX_FILENAME, TaskIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parents[1] / "lm_eval" / "tasks",
        help="Task catalogue root (default: repository lm_eval/tasks)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output file (default: ROOT/{INDEX_FILENAME})",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / INDEX_FILENAME
    index = TaskIndex.build([root])
    TaskIndex.write(index, output, root=root)
    print(f"Wrote {len(index)} task index entries to {output}")


if __name__ == "__main__":
    main()
