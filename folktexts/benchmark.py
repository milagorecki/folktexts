"""A benchmark class for measuring and evaluating LLM calibration."""

from __future__ import annotations

import dataclasses
import logging
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._io import load_json, save_json
from ._utils import get_current_timestamp, hash_dict, is_valid_number
from .acs import ACSDataset, ACSTaskMetadata
from .classifier import (
    LLMClassifier,
    TransformersLLMClassifier,
    VLLMClassifier,
    WebAPILLMClassifier,
)
from .dataset import Dataset
from .evaluation import evaluate_predictions
from .llm_utils import reasoning_to_enable_thinking
from .plotting import render_evaluation_plots, render_fairness_plots
from .prompting import (
    PROMPT_DEFAULT,
    FewShotConfig,
    PromptConfig,
    encode_row_prompt,
    encode_row_prompt_chat,
    encode_row_prompt_few_shot,
    resolve_chat_defaults,
    tokenizer_supports_system_prompt,
    tokenizer_supports_thinking,
)
from .sipp import SIPPDataset, SIPPTaskMetadata
from .task import TaskMetadata

DEFAULT_SEED: int = 42
DEFAULT_FIT_THRESHOLD_N: int = 100
DEFAULT_ROOT_RESULTS_DIR: Path = Path(".")


@dataclasses.dataclass(frozen=True, eq=True)
class BenchmarkConfig:
    """A dataclass to hold the configuration for risk-score benchmark.

    Attributes
    ----------
    numeric_risk_prompting : bool, optional
        Whether to prompt for numeric risk-estimates instead of multiple-choice
        Q&A, by default False.
    use_generated_text : bool | False, optional
        Whether to  extract answer from generated text.  Default is False.
    cot_prompting : bool, optional
        Whether to prepend a chain-of-thought instruction to the system prompt
        (before the answer-format instruction) on the generated-text path. Only
        takes effect when `use_generated_text=True`. Default is False.
    reasoning : str | None, optional
        Reasoning/thinking effort for reasoning models. Use '0' for Qwen3 to
        explicitly disable thinking; 'low'/'medium'/'high' for OpenAI reasoning
        models; a float in (0, 1] (fraction of max_new_tokens) or a positive
        integer string for Claude (literal budget_tokens, min 1024). Ignored
        for non-reasoning models. Default is None.
    few_shot_config : FewShotConfig | None, optional
        Few-shot prompting configuration (number of shots, composition, example
        order, reuse). ``None`` means zero-shot prompting.
    use_chat_template : bool, optional
        Whether to format prompts using the tokenizer's chat template, by
        default False. Only supported for local transformers models.
    chat_prompt : str | None, optional
        The assistant prefill text to use with chat templates. Defaults to
        ``PROMPT_DEFAULT``, which selects the appropriate default from the QA
        subclass (``ANTHROPIC_CHAT_PROMPT`` for MC, ``NUMERIC_CHAT_PROMPT`` for
        numeric). Pass ``None`` explicitly to disable the assistant prefill entirely.
    system_prompt : str | None, optional
        System prompt text to use with chat templates. Defaults to
        ``PROMPT_DEFAULT``, which selects the appropriate default from the QA
        subclass (``SYSTEM_PROMPT`` for MC, ``NUMERIC_SYSTEM_PROMPT`` for
        numeric). Pass ``None`` explicitly to disable the system role
        (e.g. for Gemma-style tokenizers that reject it).
    batch_size : int | None, optional
        The batch size to use for inference.
    context_size : int | None, optional
        The maximum context size when prompting the LLM.
    correct_order_bias : bool, optional
        Whether to correct the ordering bias in multiple-choice Q&A when
        prompting the LLM, by default True.
    feature_subset : list[str] | None, optional
        Whether to use a subset of the standard feature set for the task. The
        list should contain the names of the columns of features to use.
    population_filter : dict | None, optional
        Optional population filter for this benchmark; must follow the format
        `{"column_name": "value"}`.
    seed : int, optional
        Random seed -- to set for reproducibility.
    temperature : float | None, optional
        Sampling-temperature override for text-generation (chain-of-thought)
        prompting. When None (default), text generation uses greedy decoding (0.0),
        or 1.0 in thinking mode. Ignored (with a warning) for multiple-choice /
        numeric prompting, which reads untempered token probabilities.
    impute_failed_as_uniform : bool, optional
        How to handle rows whose risk score could not be parsed (NaN, e.g. a
        failed generated-text extraction or API call). When True, such scores are
        set to 0.5 (the uniform / max-entropy prior) and kept, preserving the
        sample size. When False (default), the rows are dropped from every metric.
        Either way the count is recorded as ``num_failed_scores``. Default is False.
    prompt_variation : dict | None, optional
        Dictionary of prompt style overrides (e.g. ``{"format": "bullet",
        "connector": "is"}``). ``None`` means no variation is applied.
    """

    numeric_risk_prompting: bool = False
    use_generated_text: bool = False
    cot_prompting: bool = False
    reasoning: str | None = None
    few_shot_config: FewShotConfig | None = None
    use_chat_template: bool = False
    chat_prompt: str | None = PROMPT_DEFAULT  # type: ignore[assignment]
    system_prompt: str | None = PROMPT_DEFAULT  # type: ignore[assignment]
    batch_size: int | None = None
    context_size: int | None = None
    correct_order_bias: bool = True
    feature_subset: list[str] | None = None
    population_filter: dict | None = None
    seed: int = DEFAULT_SEED
    temperature: float | None = None
    impute_failed_as_uniform: bool = False
    prompt_variation: dict | None = None

    @classmethod
    def default_config(cls, **changes):
        """Returns the default configuration with optional changes."""
        return cls(**changes)

    def update(self, **changes) -> BenchmarkConfig:
        """Update the configuration with new values."""
        possible_keys = dataclasses.asdict(self).keys()
        valid_changes = {k: v for k, v in changes.items() if k in possible_keys}

        # Log config changes
        if valid_changes:
            logging.info(f"Updating benchmark configuration with: {valid_changes}")

        # Log unused kwargs
        if len(valid_changes) < len(changes):
            unused_kwargs = {k: v for k, v in changes.items() if k not in possible_keys}
            logging.warning(f"Unused config arguments: {unused_kwargs}")

        return dataclasses.replace(self, **valid_changes)

    @classmethod
    def load_from_disk(cls, path: str | Path):
        """Load the configuration from disk (tolerant of pre-refactor JSON)."""
        obj = load_json(path)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid configuration file '{path}'.")

        # Back-compat: translate the pre-refactor flat few-shot keys into a FewShotConfig.
        legacy_n_shots = obj.pop("few_shot", None)
        legacy_reuse = obj.pop("reuse_few_shot_examples", False)
        legacy_balance = obj.pop("balance_few_shot_examples", False)
        if legacy_n_shots and obj.get("few_shot_config") is None:
            obj["few_shot_config"] = FewShotConfig(
                n_shots=legacy_n_shots,
                reuse_examples=legacy_reuse,
                compose="balanced" if legacy_balance else "random",
            )

        if isinstance(obj.get("few_shot_config"), dict):
            obj["few_shot_config"] = FewShotConfig(**obj["few_shot_config"])
        # Restore PROMPT_DEFAULT sentinel from its serialized form.
        for key in ("system_prompt", "chat_prompt"):
            if obj.get(key) == "default":
                obj[key] = PROMPT_DEFAULT

        # Drop any remaining unknown keys (removed fields, or result-file metadata)
        # so old config/result JSON still loads instead of raising TypeError.
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(obj) - valid
        if unknown:
            logging.warning(f"Ignoring unknown config keys when loading '{path}': {sorted(unknown)}")
            obj = {k: v for k, v in obj.items() if k in valid}
        return cls(**obj)

    def save_to_disk(self, path: str | Path):
        """Save the configuration to disk."""
        d = dataclasses.asdict(self)
        for key in ("system_prompt", "chat_prompt"):
            if getattr(self, key) is PROMPT_DEFAULT:
                d[key] = "default"
        save_json(d, path)

    def __hash__(self) -> int:
        """Generates a unique hash for the configuration."""
        cfg = dataclasses.asdict(self)
        for key in ("system_prompt", "chat_prompt"):
            if getattr(self, key) is PROMPT_DEFAULT:
                cfg[key] = "default"
        cfg["feature_subset"] = tuple(cfg["feature_subset"]) if cfg["feature_subset"] else None
        cfg["population_filter_hash"] = hash_dict(cfg["population_filter"]) if cfg["population_filter"] else None
        cfg["prompt_variation"] = hash_dict(cfg["prompt_variation"]) if cfg["prompt_variation"] else None
        # Hash the few-shot config deterministically. Python's builtin hash() is salted
        # (PYTHONHASHSEED), so hash(self.few_shot_config) gave result-file names a different
        # name every process; hash_dict (json-based) is stable. cfg["few_shot_config"] is
        # already the asdict form from the top-level dataclasses.asdict(self) above.
        cfg["few_shot_config"] = hash_dict(cfg["few_shot_config"]) if cfg["few_shot_config"] else None
        return int(hash_dict(cfg), 16)


class Benchmark:
    """Measures and evaluates risk scores produced by an LLM."""

    """
    Standardized configurations for the ACS data to use for benchmarking.
    """
    ACS_DATASET_CONFIGS: dict[str, Any] = {
        # ACS survey configs
        "survey_year": "2018",
        "horizon": "1-Year",
        "survey": "person",
        # Data split configs
        "test_size": 0.1,
        "val_size": 0.1,
        "subsampling": None,
        # Fixed random seed
        "seed": 42,
    }

    DATASET_CONFIGS: dict[str, Any] = {
        # survey configs should be defined in task
        # Data split configs
        "test_size": 0.1,
        "val_size": 0.1,
        "subsampling": None,
        # Fixed random seed
        "seed": 42,
    }

    def __init__(
        self,
        llm_clf: LLMClassifier,
        dataset: Dataset,
        config: BenchmarkConfig = BenchmarkConfig.default_config(),
    ):
        """A benchmark object to measure and evaluate risk scores produced by an LLM.

        Parameters
        ----------
        llm_clf : LLMClassifier
            A language model classifier object (can be local or web-hosted).
        dataset : Dataset
            The dataset object to use for the benchmark.÷
        config : BenchmarkConfig, optional
            The configuration object used to create the benchmark parameters.
            **NOTE**: This is used to uniquely identify the benchmark object for
            reproducibility; it **will not be used to change the benchmark
            behavior**. To configure the benchmark, pass a configuration object
            to the Benchmark.make_benchmark method.
        """
        self.llm_clf = llm_clf
        self.dataset = dataset
        self.config = config

        self._y_test_scores: Optional[np.ndarray] = None
        self._results_root_dir: Path = DEFAULT_ROOT_RESULTS_DIR
        self._results: Optional[dict] = None
        self._plots: Optional[dict] = None

        # Log initialization
        msg = f"\n** Benchmark initialization **\nModel: {self.model_name};\nTask: {self.task.name};\nHash: {hash(self)};\n"
        logging.info(msg)

    @property
    def configs_dict(self) -> dict:
        cnf = dataclasses.asdict(self.config)
        q = self.task.question
        if getattr(self.config, "system_prompt") is PROMPT_DEFAULT:
            cnf["system_prompt"] = "default"
            cnf["system_prompt_resolved"] = q.get_default_system_prompt()
        if getattr(self.config, "chat_prompt") is PROMPT_DEFAULT:
            cnf["chat_prompt"] = "default"
            cnf["chat_prompt_resolved"] = q.default_chat_prompt

        # Add info on model, task, and dataset
        cnf["model_name"] = self.model_name
        cnf["model_hash"] = hash(self.llm_clf)
        cnf["task_name"] = self.task.name
        cnf["task_hash"] = hash(self.task)
        cnf["dataset_name"] = self.dataset.name
        cnf["dataset_subsampling"] = self.dataset.subsampling
        cnf["dataset_hash"] = hash(self.dataset)

        return cnf

    @property
    def results(self):
        return self._results

    @property
    def task(self):
        return self.llm_clf.task

    @property
    def model_name(self):
        return self.llm_clf.model_name

    @property
    def results_root_dir(self) -> Path:
        return Path(self._results_root_dir)

    @results_root_dir.setter
    def results_root_dir(self, path: str | Path):
        self._results_root_dir = Path(path).expanduser().resolve()

    @property
    def results_dir(self) -> Path:
        """Get the results directory for this benchmark."""
        # Get subfolder name
        subfolder_name = f"{self.model_name}_bench-{hash(self)}"
        subfolder_dir_path = Path(self.results_root_dir) / subfolder_name

        # Create subfolder directory if it does not exist
        subfolder_dir_path.mkdir(exist_ok=True, parents=True)
        return subfolder_dir_path

    def __hash__(self) -> int:
        hash_params = dict(
            llm_clf_hash=hash(self.llm_clf),
            dataset_hash=hash(self.dataset),
            config_hash=hash(self.config),
        )

        return int(hash_dict(hash_params), 16)

    def _get_predictions_save_path(self, data_split: str) -> Path:
        """Standardized path to file containing predictions for the given data split."""
        assert data_split in ("train", "val", "test")
        return self.results_dir / f"{self.dataset.name}.{data_split}_predictions.csv"

    def run(
        self,
        results_root_dir: str | Path,
        fit_threshold: int | bool = 0,
        threshold_obj: str = "balanced_accuracy",
    ) -> dict:
        """Run the calibration benchmark experiment.

        Parameters
        ----------
        results_root_dir : str | Path
            Path to root directory under which results will be saved.
        fit_threshold : int | bool, optional
            Whether to fit the binarization threshold on a given number of
            training samples, by default 0 (will not fit the threshold).

        Returns
        -------
        dict
            Dictionary of evaluation results.
        """
        if self._results is not None:
            logging.warning("Benchmark was already run. Overriding previous results.")

        # Update results directory
        self.results_root_dir = Path(results_root_dir)

        # Get test data
        X_test, y_test = self.dataset.get_test()
        logging.info(f"Test data features shape: {X_test.shape}")

        # Get sensitive attribute data if available
        s_test = None
        logging.info(f"Sensitive attribute defined by task: {self.task.sensitive_attribute}")
        if self.task.sensitive_attribute is not None:
            s_test = self.dataset.get_sensitive_attribute_data().loc[y_test.index]

        # Get LLM risk-estimate predictions for each row in the test set
        test_predictions_save_path = self._get_predictions_save_path("test")
        self._y_test_scores = self.llm_clf.predict_proba(
            data=X_test,
            predictions_save_path=test_predictions_save_path,
            labels=y_test,  # used only to save alongside predictions in disk
        )
        self._y_test_scores = self.llm_clf._get_positive_class_scores(self._y_test_scores)

        # If requested, fit the threshold on a small portion of the train set
        if fit_threshold:
            if fit_threshold is True:
                fit_threshold = DEFAULT_FIT_THRESHOLD_N
            elif not is_valid_number(fit_threshold) or fit_threshold <= 0:
                raise ValueError(f"Invalid fit_threshold={fit_threshold}")

            self.llm_clf._threshold_fitted_on = fit_threshold
            self.llm_clf._threshold_obj = threshold_obj
            logging.info(f"Fitting threshold on {fit_threshold} train samples")
            X_train, y_train = self.dataset.sample_n_train_examples(fit_threshold)
            self.llm_clf.fit(
                X_train,
                y_train,
                predictions_save_path=self._get_predictions_save_path("train"),
                labels=y_train,
                threshold_obj=threshold_obj,
                impute_failed_as_uniform=self.config.impute_failed_as_uniform,
            )

        # Evaluate the test risk scores.
        # Resolve failed (NaN) risk scores per the configured policy: impute with
        # 0.5 (uniform prior, keeps the sample size) or drop them from every
        # metric. `num_failed_scores` records how many failed either way; the
        # keep-mask is all-True under imputation, so both cases share one path.
        y_true = y_test.to_numpy()
        s_arr = s_test.to_numpy() if s_test is not None else None
        self._y_test_scores, nan_mask, n_failed = self.llm_clf._apply_nan_policy(
            self._y_test_scores, self.config.impute_failed_as_uniform
        )
        n_dropped = int((~nan_mask).sum())
        if n_dropped > 0:
            logging.warning(f"Predicted scores contain NaN values, dropping {n_dropped} indices.")
            logging.info(f"Indices with NaN values: {np.where(~nan_mask)[0]}")
            logging.info(
                "New shapes:"
                f"y_test: {y_true.shape} -> {y_true[nan_mask].shape},"
                f"y_test_scores: {self._y_test_scores.shape} -> {self._y_test_scores[nan_mask].shape},"
                + ("\ns_test: " + f"{s_arr.shape} -> {s_arr[nan_mask].shape}" if s_arr is not None else "")
            )

        self._results = evaluate_predictions(
            y_true=y_true[nan_mask],
            y_pred_scores=self._y_test_scores[nan_mask],
            sensitive_attribute=s_arr[nan_mask] if s_arr is not None else None,
            threshold=self.llm_clf.threshold,
            model_name=self.llm_clf.model_name,
        )

        # Record number of *fully-failed* rows (all question permutation NaN). Partial
        # failures are hedged to 0.5 during averaging and are NOT counted here
        # (see the per-pass `_regex_*` counters for those).
        # Under the drop policy these rows are excluded (`num_samples` is the
        # survivors), under imputation they are the 0.5-imputed rows. Either way this
        # keeps the original N and the total-failure rate recoverable.
        self._results["num_failed_scores"] = n_failed

        self._results["threshold_fitted_on"] = self.llm_clf._threshold_fitted_on
        self._results["threshold_obj"] = self.llm_clf._threshold_obj if self.llm_clf._threshold_fitted_on > 0 else None
        ## TODO: set to None by default, only change when fitting, then this check is no longer needed

        if self.task.sensitive_attribute is not None:
            self._results["sensitive_attribute"] = self.task.sensitive_attribute

        # Save predictions save path
        self._results["predictions_path"] = test_predictions_save_path.as_posix()

        # Add benchmark metadata
        self._results["config"] = self.configs_dict
        self._results["benchmark_hash"] = hash(self)
        self._results["results_dir"] = self.results_dir.as_posix()
        self._results["results_root_dir"] = self.results_root_dir.as_posix()
        self._results["current_time"] = get_current_timestamp()

        # Log main results
        msg = (
            f"\n** Test results **\n"
            f"Model: {self.llm_clf.model_name};\n"
            f"\t ECE:       {self._results['ece']:.1%};\n"
            f"\t ROC AUC :  {self._results['roc_auc']:.1%};\n"
            f"\t Accuracy:  {self._results['accuracy']:.1%};\n"
            f"\t Bal. acc.: {self._results['balanced_accuracy']:.1%};\n"
        )
        logging.info(msg)

        # Render plots
        try:
            self.plot_results(show_plots=False)
        except Exception as e:
            logging.error(f"Failed to render plots: {e}")

        # Save results to disk
        self.save_results()

        return self._results

    def plot_results(self, *, show_plots: bool = True):
        """Render evaluation plots and save to disk.

        Parameters
        ----------
        show_plots : bool, optional
            Whether to show plots, by default True.

        Returns
        -------
        plots_paths : dict[str, str]
            The paths to the saved plots.
        """
        if self._results is None or self._y_test_scores is None:
            raise ValueError("No results to plot. Run the benchmark first.")

        imgs_dir = Path(self.results_dir) / "imgs"
        imgs_dir.mkdir(exist_ok=True, parents=False)
        _, y_test = self.dataset.get_test()

        plots_paths = render_evaluation_plots(
            y_true=y_test.to_numpy(),
            y_pred_scores=self._y_test_scores,
            eval_results=self.results,
            model_name=self.llm_clf.model_name,
            imgs_dir=imgs_dir,
            show_plots=show_plots,
        )

        # Plot fairness plots if sensitive attribute is provided
        if self.task.sensitive_attribute is not None:
            s_test = self.dataset.get_sensitive_attribute_data().loc[y_test.index]

            plots_paths.update(
                render_fairness_plots(
                    y_true=y_test.to_numpy(),
                    y_pred_scores=self._y_test_scores,
                    sensitive_attribute=s_test,
                    eval_results=self.results,
                    model_name=self.llm_clf.model_name,
                    group_value_map=self.task.sensitive_attribute_value_map(),
                    imgs_dir=imgs_dir,
                    show_plots=show_plots,
                )
            )

        self._results["plots"] = plots_paths

        return plots_paths

    def save_results(self, results_root_dir: str | Path | None = None):
        """Save the benchmark results to disk.

        Parameters
        ----------
        results_root_dir : str | Path, optional
            Path to root directory under which results will be saved. By default
            will use `self.results_root_dir`.
        """
        if self.results is None:
            raise ValueError("No results to save. Run the benchmark first.")

        # Update results directory if provided
        if results_root_dir is not None:
            self.results_root_dir = Path(results_root_dir)

        # Save results to disk
        results_file_name = f"results.bench-{hash(self)}.json"
        results_file_path = self.results_dir / results_file_name

        save_json(self.results, path=results_file_path)
        logging.info(f"Saved experiment results to '{results_file_path.as_posix()}'")

        # Save token usage alongside results if the classifier has a tracker
        token_tracker = getattr(self.llm_clf, "token_tracker", None)
        if token_tracker is not None:
            token_file_path = self.results_dir / f"token_usage.bench-{hash(self)}.jsonl"
            # If the tracker was streaming to a different path, copy records here too
            if token_tracker.output_path != token_file_path:
                token_tracker.save(token_file_path)
            logging.info(f"Saved token usage to '{token_file_path.as_posix()}'")

    @classmethod
    def make_acs_benchmark(
        cls,
        task_name: str,
        *,
        model: AutoModelForCausalLM | str,
        tokenizer: AutoTokenizer | None = None,
        data_dir: str | Path | None = None,
        max_api_rpm: int | None = None,
        config: BenchmarkConfig = BenchmarkConfig.default_config(),
        backend: str | None = None,
        model_name_or_path: str | Path | None = None,
        **kwargs,
    ) -> Benchmark:
        """Create a standardized calibration benchmark on ACS data.

        Parameters
        ----------
        task_name : str
            The name of the ACS task to use.
        model : AutoModelForCausalLM | str
            The transformers language model to use, or the model ID for a webAPI
            hosted model (e.g., "openai/gpt-4o-mini").
        tokenizer : AutoTokenizer, optional
            The tokenizer used to train the model (if using a transformers
            model). Not required for webAPI models.
        data_dir : str | Path, optional
            Path to the directory to load data from and save data in.
        max_api_rpm : int, optional
            The maximum number of API requests per minute for webAPI models.
        config : BenchmarkConfig, optional
            Extra benchmark configurations, by default will use
            `BenchmarkConfig.default_config()`.
        **kwargs
            Additional arguments passed to `ACSDataset` and `BenchmarkConfig`.
            By default will use a set of standardized configurations for
            reproducibility.

        Returns
        -------
        bench : Benchmark
            The ACS calibration benchmark object.
        """
        # Handle non-standard ACS arguments
        acs_dataset_configs = cls.ACS_DATASET_CONFIGS.copy()
        for arg in acs_dataset_configs:
            if arg in kwargs and kwargs[arg] != cls.ACS_DATASET_CONFIGS[arg]:
                logging.warning(
                    f"Received non-standard ACS argument '{arg}' (using "
                    f"{arg}={kwargs[arg]} instead of default {arg}={cls.ACS_DATASET_CONFIGS[arg]}). "
                    f"This may affect reproducibility."
                )
                acs_dataset_configs[arg] = kwargs.pop(arg)

        # Update config with any additional kwargs
        config = config.update(**kwargs)

        # Fetch ACS task and dataset
        use_numeric_qa = config.numeric_risk_prompting
        acs_task = ACSTaskMetadata.get_task(
            name=task_name, use_numeric_qa=use_numeric_qa, use_text_output_for_qa=config.use_generated_text
        )

        acs_dataset = ACSDataset.make_from_task(task=acs_task, cache_dir=data_dir, **acs_dataset_configs)

        return cls.make_benchmark(
            task=acs_task,
            dataset=acs_dataset,
            model=model,
            tokenizer=tokenizer,
            max_api_rpm=max_api_rpm,
            config=config,
            backend=backend,
            model_name_or_path=model_name_or_path,
        )

    @classmethod
    def make_tableshift_benchmark(
        cls,
        task_name: str,
        *,
        model: AutoModelForCausalLM | str,
        tokenizer: AutoTokenizer | None = None,
        data_dir: str | Path | None = None,
        max_api_rpm: int | None = None,
        config: BenchmarkConfig = BenchmarkConfig.default_config(),
        backend: str | None = None,
        model_name_or_path: str | Path | None = None,
        **kwargs,
    ) -> Benchmark:
        """Create a standardized calibration benchmark on TableShift BRFSS data.

        Parameters
        ----------
        task_name : str
            The name of the ACS task to use.
        model : AutoModelForCausalLM | str
            The transformers language model to use, or the model ID for a webAPI
            hosted model (e.g., "openai/gpt-4o-mini").
        tokenizer : AutoTokenizer, optional
            The tokenizer used to train the model (if using a transformers
            model). Not required for webAPI models.
        data_dir : str | Path, optional
            Path to the directory to load data from and save data in.
        max_api_rpm : int, optional
            The maximum number of API requests per minute for webAPI models.
        config : BenchmarkConfig, optional
            Extra benchmark configurations, by default will use
            `BenchmarkConfig.default_config()`.
        **kwargs
            Additional arguments passed to `TableshiftBRFSSDataset` and `BenchmarkConfig`.
            By default will use a set of standardized configurations for
            reproducibility.

        Returns
        -------
        bench : Benchmark
            The ACS calibration benchmark object.
        """
        # Handle non-standard Tableshift arguments
        tableshift_dataset_configs = cls.DATASET_CONFIGS.copy()
        for arg in tableshift_dataset_configs:
            if arg in kwargs and kwargs[arg] != cls.DATASET_CONFIGS[arg]:
                logging.warning(
                    f"Received non-standard Tableshift argument '{arg}' (using "
                    f"{arg}={kwargs[arg]} instead of default {arg}={cls.DATASET_CONFIGS[arg]}). "
                    f"This may affect reproducibility."
                )
                tableshift_dataset_configs[arg] = kwargs.pop(arg)

        # Update config with any additional kwargs
        config = config.update(**kwargs)

        # Fetch Tableshift task and dataset.
        # Imported lazily: `tableshift` is an optional, heavy dependency — avoid
        # paying its import cost on the common ACS-only path.
        from .ts import (  # noqa: PLC0415
            TableshiftBRFSSDataset,
            TableshiftBRFSSTaskMetadata,
        )

        tableshift_task = TableshiftBRFSSTaskMetadata.get_task(
            name=task_name,
            use_numeric_qa=config.numeric_risk_prompting,
            use_text_output_for_qa=config.use_generated_text,
        )

        tableshift_dataset = TableshiftBRFSSDataset.make_from_task(
            task=tableshift_task,
            cache_dir=data_dir,
            **tableshift_dataset_configs,
        )

        return cls.make_benchmark(
            task=tableshift_task,
            dataset=tableshift_dataset,
            model=model,
            tokenizer=tokenizer,
            max_api_rpm=max_api_rpm,
            config=config,
            backend=backend,
            model_name_or_path=model_name_or_path,
        )

    @classmethod
    def make_sipp_benchmark(
        cls,
        task_name: str,
        *,
        model: AutoModelForCausalLM | str,
        tokenizer: AutoTokenizer | None = None,
        data_dir: str | Path | None = None,
        max_api_rpm: int | None = None,
        config: BenchmarkConfig = BenchmarkConfig.default_config(),
        backend: str | None = None,
        model_name_or_path: str | Path | None = None,
        **kwargs,
    ) -> Benchmark:
        """Create a standardized calibration benchmark on SIPP data.

        Parameters
        ----------
        task_name : str
            The name of the ACS task to use.
        model : AutoModelForCausalLM | str
            The transformers language model to use, or the model ID for a webAPI
            hosted model (e.g., "openai/gpt-4o-mini").
        tokenizer : AutoTokenizer, optional
            The tokenizer used to train the model (if using a transformers
            model). Not required for webAPI models.
        data_dir : str | Path, optional
            Path to the directory to load data from and save data in.
        max_api_rpm : int, optional
            The maximum number of API requests per minute for webAPI models.
        config : BenchmarkConfig, optional
            Extra benchmark configurations, by default will use
            `BenchmarkConfig.default_config()`.
        **kwargs
            Additional arguments passed to `SIPPDataset` and `BenchmarkConfig`.
            By default will use a set of standardized configurations for
            reproducibility.

        Returns
        -------
        bench : Benchmark
            The calibration benchmark object.
        """
        # Handle non-standard SIPP arguments
        sipp_dataset_configs = cls.DATASET_CONFIGS.copy()
        for arg in sipp_dataset_configs:
            if arg in kwargs and kwargs[arg] != cls.DATASET_CONFIGS[arg]:
                logging.warning(
                    f"Received non-standard SIPP argument '{arg}' (using "
                    f"{arg}={kwargs[arg]} instead of default {arg}={cls.DATASET_CONFIGS[arg]}). "
                    f"This may affect reproducibility."
                )
                sipp_dataset_configs[arg] = kwargs.pop(arg)

        # Update config with any additional kwargs
        config = config.update(**kwargs)

        # Fetch SIPP task and dataset
        sipp_task = SIPPTaskMetadata.get_task(
            name=task_name,
            use_numeric_qa=config.numeric_risk_prompting,
            use_text_output_for_qa=config.use_generated_text,
        )

        sipp_dataset = SIPPDataset.make_from_task(task=sipp_task, cache_dir=data_dir, **sipp_dataset_configs)

        return cls.make_benchmark(
            task=sipp_task,
            dataset=sipp_dataset,
            model=model,
            tokenizer=tokenizer,
            max_api_rpm=max_api_rpm,
            config=config,
            backend=backend,
            model_name_or_path=model_name_or_path,
        )

    @staticmethod
    def _resolve_backend(*, backend: str | None, model) -> str:
        """Pick the inference backend for this benchmark run.

        Explicit `backend` overrides autodetection. Autodetection rules:
        - `model` is a string -> "webapi" (model ID for litellm).
        - `model` is a `vllm.LLM`-like object (has `.generate` and `.get_tokenizer`) -> "vllm".
        - Otherwise -> "transformers".
        """
        if backend is not None:
            backend = backend.lower()
            if backend not in {"transformers", "vllm", "webapi"}:
                raise ValueError(f"Unknown inference backend '{backend}'. Expected one of: 'transformers', 'vllm', 'webapi'.")
            return backend

        if isinstance(model, str):
            return "webapi"
        if hasattr(model, "generate") and hasattr(model, "get_tokenizer"):
            # Duck-typed vLLM `LLM`. transformers models also expose `.generate`,
            # but not `.get_tokenizer` — that's the discriminator.
            return "vllm"
        return "transformers"

    @staticmethod
    def _configure_task_question(task: TaskMetadata, config: BenchmarkConfig) -> None:
        """Configure the task's Q&A interface from the config.

        The Q&A interface has two independent axes — numeric vs multiple-choice
        (`numeric_risk_prompting`) and token-probability vs generated-text
        decoding (`use_generated_text`) — which together select one of the four
        concrete QA types. `TaskMetadata.get_task` returns a cached singleton, so
        both axes are set *unconditionally* here to clear any state a previous
        benchmark cell left on the task; otherwise `task.question` could return a
        stale interface.
        """
        task.use_numeric_qa = config.numeric_risk_prompting
        task.use_text_output_for_qa = config.use_generated_text
        task.cot_prompting = config.cot_prompting

    @staticmethod
    def _validate_config(config: BenchmarkConfig) -> None:
        """Reject incompatible config combinations and warn on no-op overrides."""
        if config.few_shot_config and config.use_chat_template:
            raise ValueError(
                "Cannot use both few-shot prompting and chat template formatting. Please choose one or the other."
            )

        # Reasoning/thinking is incompatible with token-probability (logprob)
        # decoding on every backend: thinking emits a trace *before* the answer,
        # so the multi-pass logprob reader (1-2 forward passes) never sees the
        # answer token, and Claude/OpenAI don't return logprobs with thinking
        # enabled anyway. Require the text-extraction path instead of shipping a
        # doomed request (which would otherwise surface as a temp!=1 / no-logprobs
        # API error deep in the call).
        if config.reasoning is not None and str(config.reasoning) != "0" and not config.use_generated_text:
            raise ValueError(
                f"Reasoning/thinking (`reasoning={config.reasoning!r}`) requires text-based "
                "answer extraction (`use_generated_text=True`): the logprob path reads "
                "untempered top-logprobs over 1-2 forward passes and cannot capture an "
                "answer that follows a thinking trace (and Claude/OpenAI don't return "
                "logprobs with thinking). Re-run with --use-generated-text."
            )

        # The chain-of-thought instruction is only appended on the generated-text
        # path (it steers free-form output the regex parser then reads); on the
        # logprob path it would be a silent no-op, so require the text path.
        if config.cot_prompting and not config.use_generated_text:
            raise ValueError(
                "Chain-of-thought prompting (`cot_prompting=True`) requires text-based "
                "answer extraction (`use_generated_text=True`): the CoT instruction only "
                "applies on the generated-text path. Re-run with --use-generated-text."
            )

        def _user_set(v) -> bool:
            # PROMPT_DEFAULT (unset) and None (explicitly disabled) are not user-provided
            # values, so they must not trigger the chat-only warning.
            return v is not None and v is not PROMPT_DEFAULT

        if not config.use_chat_template and (_user_set(config.system_prompt) or _user_set(config.chat_prompt)):
            # Warn loudly: chat-only knobs are silently ignored on the
            # zero-shot / few-shot paths, which is rarely what the user wants.
            logging.warning(
                "`system_prompt` / `chat_prompt` were provided but "
                "`use_chat_template=False`; these arguments are only used by "
                "the chat-template path and will be ignored."
            )

    @staticmethod
    def _warn_unsupported_tokenizer_features(tokenizer: AutoTokenizer, config: BenchmarkConfig) -> None:
        """Warn once, upfront, when a requested prompting feature is unavailable on this tokenizer.

        System-role support is resolved separately (and dropped) in `_build_chat_encode_row_function`.
        This helper covers the chat-template presence and `enable_thinking` support.
        """
        wants_chat = config.use_chat_template or config.reasoning is not None
        if wants_chat and getattr(tokenizer, "chat_template", None) is None:
            logging.warning(
                "Chat template / reasoning was requested, but the tokenizer has no "
                "chat_template (base model); prompts will be sent raw."
            )
            return

        supports_thinking = tokenizer_supports_thinking(tokenizer)

        # Reasoning models must have `reasoning` set explicitly: leaving it unset
        # falls back to the template default (often thinking ON), which is an
        # ambiguous, easy-to-miss state. Mirrors the web-API classifier's rule.
        if config.reasoning is None and supports_thinking:
            raise ValueError(
                "This model's chat template supports thinking (Qwen-style `enable_thinking`), "
                "so `reasoning` must be set explicitly: use '0' to disable thinking, or "
                "'low'/'medium'/'high'/a token budget to enable it. Leaving it unset would run "
                "with the template's default (often thinking on)."
            )

        if config.reasoning is not None and not supports_thinking:
            logging.warning(
                f"`reasoning={config.reasoning!r}` was requested, but this tokenizer's chat "
                "template does not accept the `enable_thinking` kwarg (the Qwen-style switch). "
                "If the model toggles thinking that way it may be ignored."
            )

    @staticmethod
    def _build_chat_encode_row_function(
        task: TaskMetadata,
        tokenizer: AutoTokenizer,
        config: BenchmarkConfig,
        prompt_config: PromptConfig,
    ) -> tuple:
        """Build the chat-template `encode_row_prompt_chat` partial, honoring tokenizer quirks.

        Returns
        -------
        tuple[Callable, PromptConfig]
            The encode function and the (possibly patched) prompt_config actually used.
        """
        if tokenizer is None:
            raise ValueError("Chat template formatting requires a local tokenizer. It is not supported for web API models.")
        logging.info("Using chat template prompting.")

        system_prompt, chat_prompt = resolve_chat_defaults(
            question=task.question,
            system_prompt=config.system_prompt,
            chat_prompt=config.chat_prompt,
        )
        # Drop the system prompt if the tokenizer's template doesn't accept
        # one (e.g. Gemma). Warn loudly when this discards a user-supplied
        # value so it isn't silently lost.
        if not tokenizer_supports_system_prompt(tokenizer):
            if config.system_prompt is not PROMPT_DEFAULT and config.system_prompt is not None:
                logging.warning(
                    "Tokenizer's chat template rejects the `system` role; "
                    "the user-supplied `system_prompt` will be discarded. "
                    "Consider folding the instruction into `custom_prompt_prefix` "
                    "or the user message instead."
                )
            else:
                logging.info("Tokenizer's chat template rejects the `system` role; running without a system prompt.")
            system_prompt = None
            # patch prompt config so the caller also gets the updated version
            prompt_config = dataclasses.replace(prompt_config, system_prompt=system_prompt)

        return partial(
            encode_row_prompt_chat,
            task=task,
            tokenizer=tokenizer,
            chat_prompt=chat_prompt,
            prompt_config=prompt_config,
            enable_thinking=reasoning_to_enable_thinking(config.reasoning),
        ), prompt_config

    @classmethod
    def _build_encode_row_function(
        cls,
        *,
        task: TaskMetadata,
        dataset: Dataset,
        tokenizer: AutoTokenizer,
        config: BenchmarkConfig,
        prompt_config: PromptConfig | None = None,
    ) -> tuple:
        """Pick the prompting function based on config (few-shot, chat-template, or zero-shot).

        Returns
        -------
        tuple[Callable, PromptConfig]
            The encode function and the prompt_config actually used (may differ from
            the input when the Gemma path drops the system prompt).
        """
        # Build PromptConfig once from the variation dict.
        if prompt_config is None:
            prompt_config = PromptConfig.from_dict(
                pv=config.prompt_variation or {},
                task=task,
                question=task.question,
                system_prompt=config.system_prompt,
            )

        logging.info(
            f"System prompt: {prompt_config.system_prompt() if prompt_config.system_prompt is not None else '(default)'}"
        )

        # Resolve tokenizer capabilities once, upfront: warn when reasoning/chat is
        # requested but unavailable, instead of discovering it per row in generation.
        if tokenizer is not None:
            cls._warn_unsupported_tokenizer_features(tokenizer, config)

        if config.few_shot_config:
            few_shot_config = config.few_shot_config
            # When correcting order bias, encode_row is called once per answer-key
            # permutation for each row. With reuse_examples=False (the default) each
            # call re-samples fresh training examples, so the averaged score conflates
            # example-selection variance with ordering variance. Force reuse so that
            # all permutations of the same row see identical few-shot context.
            if config.correct_order_bias and not few_shot_config.reuse_examples:
                logging.warning(
                    "correct_order_bias=True with reuse_examples=False: forcing "
                    "reuse_examples=True so all answer-order permutations use the "
                    "same few-shot examples."
                )
                few_shot_config = dataclasses.replace(few_shot_config, reuse_examples=True)
            logging.info(f"Using few-shot prompting (n={few_shot_config.n_shots}).")
            return partial(
                encode_row_prompt_few_shot,
                task=task,
                dataset=dataset,
                few_shot_config=few_shot_config,
                prompt_config=prompt_config,
            ), prompt_config

        # Explicit chat-template request: always use the chat encoder (it raises a
        # clear error for web-API models with no local tokenizer).
        if config.use_chat_template:
            # _build_chat_encode_row_function already returns (encode_fn, prompt_config)
            return cls._build_chat_encode_row_function(
                task=task,
                tokenizer=tokenizer,
                config=config,
                prompt_config=prompt_config,
            )

        # Generated-text (and, implicitly, reasoning) on a chat model must go
        # through the chat template so the model is cued to answer as an assistant
        # and the system prompt (format / CoT / enable_thinking) is applied there.
        # Mirrors the classifier's own `_default_encode_row`. Only when the
        # tokenizer actually has a template — base models fall through to raw prompts.
        if (config.use_generated_text or config.reasoning is not None) and getattr(
            tokenizer, "chat_template", None
        ) is not None:
            return cls._build_chat_encode_row_function(
                task=task,
                tokenizer=tokenizer,
                config=config,
                prompt_config=prompt_config,
            )

        logging.info("Using zero-shot prompting.")
        return partial(encode_row_prompt, task=task, prompt_config=prompt_config), prompt_config

    @classmethod
    def make_benchmark(
        cls,
        *,
        task: TaskMetadata | str,
        dataset: Dataset,
        model: AutoModelForCausalLM | str,
        tokenizer: AutoTokenizer | None = None,  # WebAPI models have no local tokenizer
        max_api_rpm: int | None = None,
        config: BenchmarkConfig = BenchmarkConfig.default_config(),
        backend: str | None = None,
        model_name_or_path: str | Path | None = None,
        **kwargs,
    ) -> Benchmark:
        """Create a calibration benchmark from a given configuration.

        Parameters
        ----------
        task : TaskMetadata | str
            The task metadata object or name of the task to use.
        dataset : Dataset
            The dataset to use for the benchmark.
        model : AutoModelForCausalLM | str
            The transformers language model to use, or the model ID for a webAPI
            hosted model (e.g., "openai/gpt-4o-mini").
        tokenizer : AutoTokenizer, optional
            The tokenizer used to train the model (if using a transformers
            model). Not required for webAPI models.
        max_api_rpm : int, optional
            The maximum number of API requests per minute for webAPI models.
        config : BenchmarkConfig, optional
            Extra benchmark configurations, by default will use
            `BenchmarkConfig.default_config()`.
        **kwargs
            Additional arguments for easier configuration of the benchmark.
            Will simply use these values to update the `config` object.

        Returns
        -------
        bench : Benchmark
            The calibration benchmark object.
        """
        # Update config with any additional kwargs
        config = config.update(**kwargs)

        # Validate before mutating any shared state.
        cls._validate_config(config)

        # Handle TaskMetadata object and configure its Q&A mode from the config.
        if isinstance(task, str):
            task = TaskMetadata.get_task(task)

        cls._configure_task_question(task, config)

        if config.feature_subset is not None and len(config.feature_subset) > 0:
            task = task.create_task_with_feature_subset(config.feature_subset)
            dataset.task = task

        assert isinstance(task, TaskMetadata)

        # Check dataset is compatible with task
        if dataset.task is not task and dataset.task.name != task.name:
            raise ValueError(f"Dataset task '{dataset.task.name}' does not match the provided task '{task.name}'.")

        if config.population_filter is not None:
            dataset = dataset.filter(config.population_filter)

        # Build PromptConfig once from the variation dict. Uses updated question type.
        prompt_config = PromptConfig.from_dict(
            pv=config.prompt_variation or {},
            task=task,
            question=task.question,
            system_prompt=config.system_prompt,
        )

        # Get prompting function
        encode_row_function, prompt_config = cls._build_encode_row_function(
            task=task,
            dataset=dataset,
            tokenizer=tokenizer,
            config=config,
            prompt_config=prompt_config,
        )

        ## Parse LLMClassifier parameters
        if config.temperature is not None and not config.use_generated_text:
            logging.warning(
                "`temperature` only affects text-generation (chain-of-thought) "
                "prompting; multiple-choice/numeric prompting reads untempered "
                "token probabilities, so this setting will be ignored."
            )

        llm_inference_kwargs: dict[str, Any] = {
            "correct_order_bias": config.correct_order_bias,
            "reasoning": config.reasoning,
            "prompt_config": prompt_config,  # may be patched (e.g. Gemma drops system_prompt)
            "temperature": config.temperature,
        }
        if config.batch_size is not None:
            llm_inference_kwargs["batch_size"] = config.batch_size
        if config.context_size is not None:
            llm_inference_kwargs["context_size"] = config.context_size
        if max_api_rpm is not None and isinstance(model, str):
            llm_inference_kwargs["max_api_rpm"] = max_api_rpm

        # Create LLMClassifier object
        resolved_backend = cls._resolve_backend(backend=backend, model=model)

        if resolved_backend == "webapi":
            llm_clf = WebAPILLMClassifier(
                model_name=model,
                task=task,
                encode_row=encode_row_function,
                **llm_inference_kwargs,
            )
            logging.info(f"Using webAPI model: {model}")

        elif resolved_backend == "vllm":
            llm_clf = VLLMClassifier(
                llm=model,
                tokenizer=tokenizer,
                task=task,
                model_name_or_path=model_name_or_path,
                encode_row=encode_row_function,
                **llm_inference_kwargs,
            )
            logging.info(f"Using local vLLM model: {llm_clf.model_name}")

        else:  # transformers
            assert tokenizer is not None, "Tokenizer must be provided for local transformers models."
            llm_clf = TransformersLLMClassifier(
                model=model,
                tokenizer=tokenizer,
                task=task,
                encode_row=encode_row_function,
                **llm_inference_kwargs,
            )
            logging.info(f"Using local transformers model: {llm_clf.model_name}")

        logging.info("Exemplary row encoding")
        logging.info(
            # reuse_examples=True avoids advancing the dataset's RNG at construction
            # time, which would shift the random stream for all subsequent few-shot
            # example draws and threshold fitting.
            llm_clf.encode_row(dataset.sample_n_train_examples(n=1, reuse_examples=True)[0].iloc[0])
        )

        return cls(
            llm_clf=llm_clf,
            dataset=dataset,
            config=config,
        )
