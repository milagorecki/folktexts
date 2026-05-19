#!/usr/bin/env python3
"""Runs the LLM calibration benchmark from the command line.
Exemplary Usage:
    run_benchmark --model gpt2 --results-dir './results/test/' --data-dir '../llm_fairness/folktexts/data' --task ACSIncome --subsampling 0.01 --variation "format=bullet,connector=is" --logger-level ERROR
"""  # noqa: E501

import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import dotenv

from folktexts._utils import ParseDict
from folktexts.llm_utils import get_model_folder_path
from folktexts.prompting import DEFAULT_PROMPT_STYLE

DEFAULT_ACS_TASK = "ACSIncome"
ACS_TASKS = (
    "ACSIncome",
    "ACSEmployment",
    "ACSMobility",
    "ACSTravelTime",
    "ACSPublicCoverage",
    "ACSHealthInsurance",
    "ACSIncomePovertyRatio",
)
TABLESHIFT_TASKS = (
    "BRFSS_Diabetes",
    "BRFSS_Blood_Pressure",
)
SIPP_TASKS = ("SIPP",)

DEFAULT_BATCH_SIZE = 16
DEFAULT_CONTEXT_SIZE = 600
DEFAULT_SEED = 42


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(description="Benchmark risk scores produced by a language model on ACS data.")

    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(",")

    # List of command-line arguments, with type and helper string
    cli_args: list[Any] = [
        ("--model", str, "[str] Model name or path to model saved on disk"),
        ("--results-dir", str, "[str] Directory under which this experiment's results will be saved"),
        ("--data-dir", str, "[str] Root folder to find datasets on"),
        ("--task", str, "[str] Name of the task to run the experiment on", False, DEFAULT_ACS_TASK),
        ("--few-shot", int, "[int] Use few-shot prompting with the given number of shots", False),
        ("--batch-size", int, "[int] The batch size to use for inference", False, DEFAULT_BATCH_SIZE),
        ("--context-size", int, "[int] The maximum context size when prompting the LLM", False, DEFAULT_CONTEXT_SIZE),
        ("--fit-threshold", int, "[int] Whether to fit the prediction threshold, and on how many samples", False),
        ("--subsampling", float, "[float] Which fraction of the dataset to use (if omitted will use all data)", False),
        ("--seed", int, "[int] Random seed -- to set for reproducibility", False, DEFAULT_SEED),
    ]

    for arg in cli_args:
        parser.add_argument(  # type: ignore[arg-type]
            arg[0],
            type=arg[1],
            help=arg[2],
            required=(arg[3] if len(arg) > 3 else True),  # NOTE: required by default
            default=(arg[4] if len(arg) > 4 else None),  # default value if provided
        )

    # Add special arguments (e.g., boolean flags or multiple-choice args)
    parser.add_argument(
        "--use-web-api-model",
        help="[bool] Whether use a model hosted on a web API (instead of a local model)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        help="[string] Directory under which models are saved.",
        required=False,
    )

    parser.add_argument(
        "--dont-correct-order-bias",
        help="[bool] Whether to avoid correcting ordering bias, by default will correct it",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use-generated-text",
        # type = bool,
        help="[bool] Whether to extract answers from generated text",
        # required=False,
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--numeric-risk-prompting",
        help="[bool] Whether to prompt for numeric risk-estimates instead of multiple-choice Q&A",
        action="store_true",
        default=False,
    )

    def _int_or_float_or_str(value: str) -> int | float | str:
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    parser.add_argument(
        "--reasoning",
        help=(
            "[str|int] Reasoning/thinking effort for reasoning models. "
            "Use '0' for Qwen3 to explicitly disable thinking (omit to use model default). "
            "Use 'low', 'medium', or 'high' for OpenAI reasoning models (sets reasoning_effort). "
            "Use a float in (0, 1] (fraction of max_new_tokens) or a positive integer "
            "(literal budget_tokens, min 1024) for Claude reasoning models. "
            "Ignored for non-reasoning models. Required for known reasoning models."
        ),
        type=_int_or_float_or_str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--reuse-few-shot-examples",
        help="[bool] Whether to reuse the same samples for few-shot prompting (or sample new ones every time)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--compose-few-shot-examples",
        help=(
            "[str|list] How to select samples in few-shot prompting: random, balanced or list of speicified "
            "class counts. Defaults to random."
        ),
        default="random",
        required=False,
    )

    parser.add_argument(
        "--example-order",
        type=str,
        help=(
            "[str] Comma-separated permutation of few-shot example indices, e.g. '2,0,1'. "
            "Only used when --few-shot is set."
        ),
        required=False,
        default=None,
    )

    # Optionally, receive a list of features to use (subset of original list)
    parser.add_argument(
        "--use-feature-subset",
        type=list_of_strings,
        help="[str] Optional subset of features to use for prediction, comma separated",
        required=False,
    )

    parser.add_argument(
        "--use-population-filter",
        type=list_of_strings,
        help=(
            "[str] Optional population filter for this benchmark; must follow "
            "the format 'column_name=value' to filter the dataset by a specific value."
        ),
        required=False,
    )

    parser.add_argument(
        "--max-api-rpm",
        type=int,
        help="[int] Maximum number of API requests per minute (if using a web-hosted model)",
        required=False,
    )

    parser.add_argument(
        "--logger-level",
        type=str,
        help="[str] The logging level to use for the experiment",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        required=False,
        default="WARNING",
    )

    parser.add_argument(
        "--threshold-obj",
        type=str,
        help="[str] The objective to use for fitting the threshold",
        choices=["accuracy", "balanced_accuracy"],
        required=False,
        default="balanced_accuracy",
    )

    parser.add_argument(
        "--variation",
        help="[dict] Dictionary specifying variations of data point serialization.",
        nargs="*",
        action=ParseDict,
        required=False,
        default={},
    )

    return parser


def main():
    """Prepare and launch the LLM-as-classifier experiment using ACS or Tableshift data."""

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.logger_level)
    pretty_args_str = json.dumps(vars(args), indent=4, sort_keys=True)
    logging.info(f"Current python executable: '{sys.executable}'")
    logging.info(f"Received the following cmd-line args: {pretty_args_str}")

    dotenv_succes = dotenv.load_dotenv()
    logging.debug(f"dotenv.load_dotenv() returned {dotenv_succes}.")

    # Parse population filter if provided
    population_filter_dict = None
    if args.use_population_filter:
        from folktexts.cli._utils import cmd_line_args_to_kwargs

        population_filter_dict = cmd_line_args_to_kwargs(args.use_population_filter)

    prompt_variation_dict = DEFAULT_PROMPT_STYLE
    if args.variation != {}:
        # update with args.variation
        prompt_variation_dict = {**prompt_variation_dict, **args.variation}

    # Load model and tokenizer
    # > Web-hosted LLM
    # Reasoning requires text-based answer extraction
    if args.reasoning is not None and not args.use_generated_text:
        parser.error("--use-generated-text must be set when --reasoning is specified.")

    if args.use_web_api_model:
        model = args.model
        tokenizer = None
    # > Local LLM
    else:
        from folktexts.llm_utils import load_model_tokenizer

        model_path = args.model
        if args.models_dir:
            model_path = get_model_folder_path(args.model, root_dir=args.models_dir)
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model folder not found at '{model_path}'.")
        if args.use_generated_text:
            logging.info("Tokenizer padding_side set to 'left'.")
            model, tokenizer = load_model_tokenizer(args.model, padding_side="left")
        else:
            model, tokenizer = load_model_tokenizer(args.model)

    # Build FewShotConfig if few-shot prompting is requested
    from folktexts.benchmark import BenchmarkConfig
    from folktexts.prompting import FewShotConfig

    few_shot_config = None
    if args.few_shot:
        few_shot_config = FewShotConfig(
            n_shots=args.few_shot,
            compose=args.compose_few_shot_examples,
            reuse_examples=args.reuse_few_shot_examples,
            example_order=args.example_order,
        )

    # Fill Benchmark config
    config = BenchmarkConfig(
        few_shot_config=few_shot_config,
        use_generated_text=args.use_generated_text,
        numeric_risk_prompting=args.numeric_risk_prompting,
        reasoning=args.reasoning,
        batch_size=args.batch_size,
        context_size=args.context_size,
        correct_order_bias=not args.dont_correct_order_bias,
        feature_subset=args.use_feature_subset or None,
        population_filter=population_filter_dict,
        seed=args.seed,
        prompt_variation=prompt_variation_dict,
    )

    # Create Benchmark object
    from folktexts.benchmark import Benchmark

    task = args.task
    benchmark_fun_dict = {
        "acs": Benchmark.make_acs_benchmark,
        "tableshift": Benchmark.make_tableshift_benchmark,
        "sipp": Benchmark.make_sipp_benchmark,
    }
    if task in ACS_TASKS:
        benchmark_fun = benchmark_fun_dict["acs"]
    elif task in TABLESHIFT_TASKS:
        benchmark_fun = benchmark_fun_dict["tableshift"]
    elif task in SIPP_TASKS:
        benchmark_fun = benchmark_fun_dict["sipp"]
    else:
        raise ValueError(f"Unknown task name: {args.task}")
    bench = benchmark_fun(
        task_name=args.task,
        model=model,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        config=config,
        subsampling=args.subsampling,
        max_api_rpm=args.max_api_rpm,
    )

    # Set-up results directory
    from folktexts.cli._utils import get_or_create_results_dir

    results_dir = get_or_create_results_dir(
        model_name=Path(args.model).name,
        task_name=bench.task.name,
        results_root_dir=args.results_dir,
    )
    logging.info(f"Saving results to {results_dir.as_posix()}")

    # Run benchmark
    bench.run(
        results_root_dir=results_dir,
        fit_threshold=args.fit_threshold,
        threshold_obj=args.threshold_obj,
    )
    bench.save_results()

    # Save results
    import pprint

    pprint.pprint(bench.results, indent=4, sort_dicts=True)

    # Finish
    from folktexts._utils import get_current_timestamp

    print(f"\nFinished experiment successfully at {get_current_timestamp()}\n")


if __name__ == "__main__":
    main()
