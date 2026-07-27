#!/usr/bin/env python3
"""Re-run a single experiment from its saved JSON config.

Used both to re-run an experiment locally and as the HTCondor job entrypoint
(via ``scripts/htcondor_job.sh``) when an experiment is launched with
``wrap_job=True``.
"""

import sys
from argparse import ArgumentParser
from subprocess import call
from typing import Any

from folktexts._io import load_json

from .experiments import Experiment, encode_experiment_args


def setup_arg_parser() -> ArgumentParser:
    # Init parser
    parser = ArgumentParser(description="Re-run a single experiment from its JSON config file.")
    parser.add_argument(
        "--experiment-json",
        type=str,
        help="[string] Path to an experiment JSON file to load.",
        required=True,
    )
    # TODO: add over-writable key-word arguments

    return parser


if __name__ == "__main__":
    # Parse command-line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Load experiment from JSON file
    print(f"Running experiment from config file at '{args.experiment_json}'...")
    data: Any = load_json(args.experiment_json)
    exp = Experiment(**data)

    # Reconstruct the run_benchmark command with the same encoding used for
    # HTCondor submission (bool -> bare flag, whitespace values -> b64), then run it.
    cmdline_args = encode_experiment_args(exp.kwargs)
    raise SystemExit(call([sys.executable, exp.executable_path, *cmdline_args]))
