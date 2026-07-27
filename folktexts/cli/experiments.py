"""General constants and helper classes to run the main experiments on htcondor."""

import base64
import collections
import collections.abc
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import htcondor

from folktexts._utils import hash_dict

# monkey patch for classad
collections.MutableMapping = collections.abc.MutableMapping  # type: ignore[attr-defined]

import classad  # noqa: E402

# Cluster settings
DEFAULT_JOB_BID = 25  # htcondor bid (min. is 15 apparently...)
DEFAULT_JOB_CPUS = 4  # number of CPUs per experiment (per cluster job)
DEFAULT_JOB_MEMORY_GB = 62  # GBs of memory
DEFAULT_GPU_MEMORY_GB = 30  # GBs of GPU memory

MAX_RUNNING_PRICE = 1500  # Max price for running a job

# Execute nodes to avoid (e.g. GPUs where cuDNN SDPA crashes) are listed, one
# short hostname per line, in this plain-text file — edit it to add/remove nodes
# without touching code. Applied as an HTCondor `requirements` clause so jobs
# never land on them.
EXCLUDED_NODES_FILE = Path(__file__).resolve().parents[2] / "excluded_nodes.txt"


def get_excluded_nodes() -> list[str]:
    """Short-hostnames of execute nodes to avoid.

    Read from the excluded-nodes file (one hostname per line; ``#`` starts a
    comment; path overridable via ``FOLKTEXTS_EXCLUDED_NODES_FILE``), plus any
    from the ``FOLKTEXTS_EXCLUDED_NODES`` env var (comma-separated) for one-offs.
    """
    nodes: list[str] = []
    path = Path(os.getenv("FOLKTEXTS_EXCLUDED_NODES_FILE") or EXCLUDED_NODES_FILE)
    if path.exists():
        for raw in path.read_text().splitlines():
            entry = raw.split("#", 1)[0].strip()  # strip inline/full-line comments
            if entry:
                nodes.append(entry)
    nodes += [n.strip() for n in os.getenv("FOLKTEXTS_EXCLUDED_NODES", "").split(",") if n.strip()]
    return list(dict.fromkeys(nodes))  # dedupe, preserve order


def excluded_nodes_requirement(nodes: list[str]) -> str:
    """Build an HTCondor `requirements` clause that excludes `nodes` (matched by
    short hostname, i.e. up to the first dot of `Machine`). Empty when no nodes."""
    if not nodes:
        return ""
    pattern = "|".join(re.escape(n) for n in nodes)
    # regexp() is true on an excluded machine; `=!= true` keeps only the others
    # (and is undefined-safe). `([.]|$)` anchors to the short hostname.
    return f'(regexp("^({pattern})([.]|$)", TARGET.Machine) =!= true)'


@dataclass
class Experiment:
    """A generic experiment to run on the cluster."""

    executable_path: str
    env_vars: str = ""
    kwargs: dict = field(default_factory=dict)

    job_cpus: int = DEFAULT_JOB_CPUS
    job_gpus: int = 0
    job_memory_gb: int = DEFAULT_JOB_MEMORY_GB
    job_gpu_memory_gb: int = DEFAULT_GPU_MEMORY_GB
    job_bid: int = DEFAULT_JOB_BID

    # Human-readable job name. Sets `JobBatchName` (shown in `condor_q -batch`),
    # and — when `wrap_job` is on — also names the job's executable so it appears
    # in the default `condor_q` CMD column instead of the wrapper's filename.
    job_name: str = ""

    # When True, submit via a stable wrapper script that runs the experiment from
    # its saved JSON config, so `condor_q` shows just the job name instead of the
    # full `python run_benchmark.py <flags>` command. Default keeps direct submission.
    wrap_job: bool = False

    _all_experiments: ClassVar[list["Experiment"]] = []

    def __post_init__(self):
        # Add experiment to the class-level list
        self._all_experiments.append(self)

    @classmethod
    def get_all_experiments(cls):
        return cls._all_experiments

    def __getattr__(self, name: str):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            raise AttributeError(f"Attribute '{name}' not found in Experiment.")

    def hash(self) -> str:
        """Generate a hexadecimal hash that uniquely identifies the experiment's arguments."""
        # Get hash of the experiment's arguments
        kwargs_for_hash = dict(
            executable_path=self.executable_path,
            **self.kwargs,
        )

        # These kwargs shouldn't be used to generate a unique hash
        kwargs_for_hash.pop("results_dir", None)
        kwargs_for_hash.pop("hash", None)

        return hash_dict(kwargs_for_hash)

    def to_dict(self) -> dict:
        return asdict(self)


# Wrapper submitted to HTCondor when `wrap_job` is set. It runs one experiment
# from its saved JSON config (path passed via the FOLKTEXTS_EXP_JSON env var), so
# `condor_q` shows just this script instead of the full flag list.
JOB_WRAPPER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "htcondor_job.sh"


def encode_experiment_args(kwargs: dict) -> list[str]:
    """Encode experiment kwargs into ``run_benchmark`` command-line tokens.

    - ``bool`` True -> bare ``--flag`` (argparse ``store_true``); False -> omitted.
    - ``str`` with whitespace -> ``--flag=b64:<base64>``. HTCondor's Submit class
      cannot pass argument values containing spaces regardless of quoting, so such
      values are base64-encoded; ``run_benchmark`` decodes ``b64:`` transparently.
    - everything else -> ``--flag=<value>``.
    """
    tokens: list[str] = []
    for key, value in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                tokens.append(flag)
        elif isinstance(value, str) and " " in value:
            tokens.append(f"{flag}=b64:" + base64.b64encode(value.encode()).decode())
        else:
            tokens.append(f"{flag}={value}")
    return tokens


def launch_experiment_job(exp: Experiment):

    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = (Path(exp.results_dir) / f"log.$(Cluster).$(Process).exp{exp.hash()}").as_posix()

    if exp.wrap_job:
        # Run through the wrapper: `condor_q` shows the job name instead of the
        # full command. The experiment config travels via the environment (its
        # saved JSON path), so the shared wrapper is read-only and never rewritten.
        exp_json = (Path(exp.results_dir) / f"experiment.{exp.hash()}.json").as_posix()
        env_parts = [f"FOLKTEXTS_PYTHON={sys.executable}", f"FOLKTEXTS_EXP_JSON={exp_json}"]
        if exp.env_vars:
            env_parts.append(exp.env_vars)
        arguments = ""
        environment = ";".join(env_parts)

        # `condor_q` shows basename(executable). To display the job name there,
        # point the executable at a per-job symlink (in this job's own results_dir,
        # so concurrent jobs never collide) that targets the shared wrapper.
        executable = JOB_WRAPPER_PATH.as_posix()
        if exp.job_name:
            link = Path(exp.results_dir) / exp.job_name.replace("/", "-").replace(" ", "-")
            try:
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(JOB_WRAPPER_PATH)
                executable = link.as_posix()
            except OSError as err:
                logging.warning(f"Could not create job-name symlink '{link}': {err}; using the wrapper directly.")
    else:
        # Direct submission: `python run_benchmark.py <flags>` (full command
        # visible in `condor_q`).
        executable = sys.executable
        arguments = f"{exp.executable_path} " + " ".join(encode_experiment_args(exp.kwargs))
        environment = exp.env_vars or ""

    requirement_clauses = []
    if exp.job_gpus > 0:
        requirement_clauses.append(f"(TARGET.CUDAGlobalMemoryMb > {exp.job_gpu_memory_gb * 1_000})")
    excluded_clause = excluded_nodes_requirement(get_excluded_nodes())
    if excluded_clause:
        requirement_clauses.append(excluded_clause)
    requirements = " && ".join(requirement_clauses)

    # Construct job description
    job_description = htcondor.Submit(
        {
            "executable": executable,
            "arguments": arguments,
            "output": f"{cluster_job_log_name}.out",
            "error": f"{cluster_job_log_name}.err",
            "log": f"{cluster_job_log_name}.log",
            "request_cpus": f"{exp.job_cpus}",
            "request_gpus": f"{exp.job_gpus}",
            "request_memory": f"{exp.job_memory_gb}GB",
            "request_disk": "10GB",
            "jobprio": f"{exp.job_bid - 1000}",
            "notify_user": "",
            "notification": "error",
            # Inherit the toolchain-relevant vars from the submit environment so
            # the job's PATH includes the system bins. Without this, HTCondor gives
            # the job a stripped-down PATH, and vLLM/Triton's runtime kernel build
            # (gcc -> collect2 -> ld) fails with "cannot find 'ld'" even though ld
            # is present on the execute node. Selective (not `getenv = True`) to
            # avoid overriding HTCondor's runtime-assigned CUDA_VISIBLE_DEVICES.
            "getenv": "PATH LD_LIBRARY_PATH LIBRARY_PATH CPATH HOME CONDA_PREFIX",
            # Environment variables (applied on top of the inherited ones)
            "environment": environment,
            # GPU requirements
            "requirements": requirements,
            # Concurrency limits:
            # > each job uses this amount of resources out of a pool of 10k
            "concurrency_limits": "user.folktexts:100",  # 100 jobs in parallel
            "+MaxRunningPrice": MAX_RUNNING_PRICE,
            "+RunningPriceExceededAction": classad.quote("restart"),
        }
    )

    # Also set JobBatchName so `condor_q -batch` groups/labels by the job name.
    if exp.job_name:
        job_description["batch_name"] = exp.job_name

    # Submit job to the htcondor scheduler
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job_description)

    logging.info(f"Launched {submit_result.num_procs()} processe(s) with cluster-ID={submit_result.cluster()}\n")

    return submit_result
