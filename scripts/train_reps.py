from __future__ import annotations

from typing import Sequence

try:
    from scripts.policy_search_training_cli import (
        build_training_arg_parser,
        default_output_dir_for_algorithm,
        main_for_algorithm,
    )
except ModuleNotFoundError:
    from policy_search_training_cli import (
        build_training_arg_parser,
        default_output_dir_for_algorithm,
        main_for_algorithm,
    )


def build_arg_parser():
    return build_training_arg_parser(algorithm="reps")


def _default_output_dir_for_algorithm(algorithm: str):
    return default_output_dir_for_algorithm(algorithm)


def main(argv: Sequence[str] | None = None) -> int:
    return main_for_algorithm(algorithm="reps", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
