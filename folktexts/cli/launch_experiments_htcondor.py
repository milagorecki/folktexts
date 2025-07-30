#!/usr/bin/env python3
"""Launch htcondor jobs for all benchmark experiments.
Usage:
    - ACS: python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_benchmark.py --results-dir './results/test/' --logger-level=INFO --task ACSIncome --model openai-community/gpt2 --subsampling=0.01  --variation="format=bullet;connector=is"
    - TableShift: python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_benchmark.py --results-dir './results/test/' --logger-level=INFO --task BRFSS_Diabetes --model openai-community/gpt2 --subsampling=0.01  --variation="format=bullet;connector=is"
    - web-API: python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_benchmark.py --results-dir '/fast/mgorecki/monoculture/results/folktexts/0-bullet-is/' --logger-level=INFO --task=ACSIncome --variation="format=bullet;connector=is" --model=gpt-4.1  --use-web-api-model --environment="AZURE_API_BASE=$AZURE_API_BASE;AZURE_API_KEY=$AZURE_API_KEY"
    - few-shot: python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_benchmark.py --results-dir '/fast/mgorecki/monoculture/results/folktexts/10-variations/' --logger-level=INFO --models-dir /fast/mgorecki/models/ --task=ACSIncome --fit-threshold=2000 --variation="format=comma;connector=:;granularity=original;order=RAC1P,SEX,WKHP,RELP,POBP,OCCP,MAR,SCHL,COW,AGEP" --subsampling=0.01 --model openai-community/gpt2 --reuse-few-shot-examples=True --few-shot=10 --balance-few-shot-examples=True
"""
import argparse
import math
from pathlib import Path
from pprint import pprint

from folktexts._io import load_json, save_json
from folktexts.llm_utils import get_model_folder_path, get_model_size_B
from folktexts.cli._utils import get_or_create_results_dir

from .experiments import Experiment, launch_experiment_job
import logging

# All ACS prediction tasks
ACS_TASKS = (
    "ACSIncome",
    "ACSEmployment",
    "ACSMobility",
    "ACSTravelTime",
    "ACSPublicCoverage",
)
TABLESHIFT_TASKS = (
    # "BRFSS_Diabetes",
    "BRFSS_Blood_Pressure",
)
SIPP_TASKS = ("SIPP",)
TASKS = ACS_TASKS + TABLESHIFT_TASKS + SIPP_TASKS
################
# Useful paths #
################
ROOT_DIR = Path("/fast/groups/sf")
# ROOT_DIR = Path("~").expanduser().resolve()               # on local machine

# data directory
ACS_DATA_DIR = ROOT_DIR / "data"
TABLESHIFT_DATA_DIR = Path("/fast/mgorecki/monoculture/data")

# Models save directory
DEFAULT_MODELS_DIR = ROOT_DIR / "huggingface-models"  # Path('/fast/rolmedo/') / 'models'


##################
# Global configs #
##################
BATCH_SIZE = 10
CONTEXT_SIZE = 750

JOB_CPUS = 4
JOB_MEMORY_GB = 60
JOB_BID = 500

# LLMs to evaluate
LLM_MODELS = [
    # Google Gemma2 models
    "google/gemma-2b",
    "google/gemma-1.1-2b-it",
    "google/gemma-7b",
    "google/gemma-1.1-7b-it",

    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",

    # Meta Llama3 models
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",

    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",

    "meta-llama/Meta-Llama-3.2-1B",
    "meta-llama/Meta-Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3.2-3B",
    "meta-llama/Meta-Llama-3.2-3B-Instruct",

    "meta-llama/Meta-Llama-3.3-70B-Instruct",

    # Mistral AI models
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",

    "mistralai/Mistral-Small-24B-Base-2501",
    "mistralai/Mistral-Small-24B-Instruct-2501",

    # Yi models
    "01-ai/Yi-6B",
    "01-ai/Yi-6B-Chat",
    "01-ai/Yi-34B",
    "01-ai/Yi-34B-Chat",

    # Qwen2 models
    # "Qwen/Qwen2-1.5B",
    # "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-72B",
    "Qwen/Qwen2-72B-Instruct",

    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct",

    # OLMo models
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-1B-hf",
    "allenai/OLMo-7B-0724-hf",
    "allenai--OLMo-7B-hf",
    "allenai--OLMo-7B-Instruct-hf",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-Instruct",
]


# Function that defines common settings among all LLM-as-clf experiments
def make_llm_clf_experiment(
    executable_path: str,
    model_name: str,
    task: str,
    results_dir: str,
    models_dir: str,
    env_vars_str: str = "",
    **kwargs,
) -> Experiment:
    """Create an experiment object to run.
    """
    # Get model size
    model_size_B = get_model_size_B(model_name, default=8)

    # Get model path
    model_path = get_model_folder_path(model_name, root_dir=models_dir)
    if not Path(model_path).exists() and "use_web_api_model" not in kwargs:
        raise FileNotFoundError(f"Model folder not found at '{model_path}'.")

    # Split experiment and job kwargs
    job_kwargs = {key: val for key, val in kwargs.items() if key.startswith("job_")}
    experiment_kwargs = {key: val for key, val in kwargs.items() if key not in job_kwargs}

    # Set default job kwargs
    job_kwargs.setdefault("job_cpus", JOB_CPUS)
    job_kwargs.setdefault("job_gpus", math.ceil(model_size_B / 40))     # One GPU per 40B parameters
    job_kwargs.setdefault("job_memory_gb", JOB_MEMORY_GB)
    job_kwargs.setdefault("job_gpu_memory_gb", 35 if model_size_B < 5 else 60)
    job_kwargs.setdefault("job_bid", JOB_BID)

    # Set default experiment kwargs
    n_shots = int(experiment_kwargs.get("few_shot", 1))
    experiment_kwargs.setdefault("batch_size", math.ceil(BATCH_SIZE / n_shots))
    experiment_kwargs.setdefault("context_size", CONTEXT_SIZE * n_shots)
    experiment_kwargs.setdefault("data_dir", ACS_DATA_DIR.as_posix() if task in ACS_TASKS else TABLESHIFT_DATA_DIR.as_posix())
    # experiment_kwargs.setdefault("fit_threshold", FIT_THRESHOLD)

    if "use_feature_subset" in kwargs:
        feature_subset = kwargs["use_feature_subset"].split(",")
        task_with_subset = f"{task}_" + "_".join(sorted(feature_subset))

    results_dir = get_or_create_results_dir(
        model_name=model_name,
        task_name=task if "use_feature_subset" not in kwargs else task_with_subset,
        results_root_dir=results_dir,
    )
    logging.info(f"Updated results_dir to {results_dir.as_posix()}")

    # Define experiment
    exp = Experiment(
        executable_path=executable_path,
        env_vars=env_vars_str,
        kwargs=dict(
            model=model_path if "use_web_api_model" not in kwargs else model_name,
            task=task,
            results_dir=results_dir.as_posix(),
            **experiment_kwargs,
        ),
        **job_kwargs,
    )

    # Create LLM results directory
    save_json(
        obj=exp.to_dict(),
        path=Path(results_dir) / f"experiment.{exp.hash()}.json",
        overwrite=True,
    )
    logging.info(f"Created experiment.{exp.hash()}.json at {results_dir.as_posix()}")
    print(f"Saving experiment json to {results_dir.as_posix()}")

    return exp


def setup_arg_parser() -> argparse.ArgumentParser:
    # Init parser
    parser = argparse.ArgumentParser(description="Launch experiments to evaluate LLMs as classifiers.")

    parser.add_argument(
        "--executable-path",
        type=str,
        help="[string] Path to the executable script to run.",
        required=True,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        help="[string] Directory under which results will be saved.",
        required=True,
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        help="[string] Directory under which models are saved.",
        required=False,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="[string] Model name on huggingface hub - can provide multiple!",
        required=False,
        action="append",
    )

    parser.add_argument(
        "--task",
        type=str,
        help="[string] ACS/Tableshift task name to run experiments on - can provide multiple!",
        required=False,
        action="append",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct folder structure and print experiments without launching them.",
        default=False,
    )

    parser.add_argument(
        "--experiment-json",
        type=str,
        help="[string] Path to an experiment JSON file to load. Will override all other args.",
        required=False,
    )

    parser.add_argument(
        "--environment",
        type=str,
        help=(
            "[string] String defining environment variables to be passed to "
            "launched jobs, in the form 'VAR1=val1;VAR2=val2;...'."
        ),
        required=False,
    )

    return parser


def main():
    # Parse command-line arguments
    parser = setup_arg_parser()
    args, extra_kwargs = parser.parse_known_args()

    # Parse extra kwargs
    from ._utils import cmd_line_args_to_kwargs
    extra_kwargs = cmd_line_args_to_kwargs(extra_kwargs)

    # Prepare command-line arguments
    models_dir = DEFAULT_MODELS_DIR if not args.models_dir else args.models_dir
    models = args.model or LLM_MODELS
    tasks = args.task or TASKS
    executable_path = Path(args.executable_path).resolve()
    if not executable_path.exists() or not executable_path.is_file():
        raise FileNotFoundError(f"Executable script not found at '{executable_path}'.")

    # Set-up results directory
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    print(f"Source for model files set to '{models_dir}'")

    # Load experiment from JSON file if provided
    if args.experiment_json:
        print(f"Launching job for experiment at '{args.experiment_json}'...")
        exp = Experiment(**load_json(args.experiment_json))
        all_experiments = [exp]

    # Otherwise, run all experiments planned
    else:
        all_experiments = [
            make_llm_clf_experiment(
                executable_path=executable_path.as_posix(),
                model_name=model,
                task=task,
                results_dir=args.results_dir,
                models_dir=models_dir,
                env_vars_str=args.environment,
                **extra_kwargs,
            )
            for model in models
            for task in tasks
        ]

    # Log total number of experiments
    print(f"Launching {len(all_experiments)} experiment(s)...\n")
    for i, exp in enumerate(all_experiments):
        cluster_id = launch_experiment_job(exp).cluster() if not args.dry_run else None
        print(f"{i:2}. cluster-id={cluster_id}")
        pprint(exp.to_dict(), indent=4)


if __name__ == "__main__":
    main()
