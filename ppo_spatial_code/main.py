from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair-RC-RL command entry.")
    parser.add_argument("command", choices=["train", "inference", "eval"])
    args, remaining = parser.parse_known_args()

    if args.command == "train":
        if __package__:
            from . import train
        else:  # pragma: no cover - supports direct script execution
            import train

        train.main(remaining)
    else:
        if __package__:
            from . import inference
        else:  # pragma: no cover - supports direct script execution
            import inference

        inference.main(remaining)


if __name__ == "__main__":
    main()
