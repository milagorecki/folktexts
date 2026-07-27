"""Module containing the base class for all LLM risk classifiers."""

from __future__ import annotations

import dataclasses
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from os import getenv, remove
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm.auto import tqdm

from folktexts.dataset import Dataset
from folktexts.evaluation import compute_best_threshold
from folktexts.prompting import PromptConfig
from folktexts.prompting import encode_row_prompt as default_encode_row_prompt
from folktexts.qa_interface import DirectNumericQA, GeneratedTextQA, MultipleChoiceQA
from folktexts.task import TaskMetadata

from .._utils import hash_dict, hash_function


class EncodeRowCallable(Protocol):
    def __call__(self, row: pd.Series, **kwargs) -> str: ...


DEFAULT_CONTEXT_SIZE = 600
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 42
# Single source of truth for the generated-text output budget: the QA type owns
# it, and `run_benchmark` sizes vLLM's `max_model_len` from the same value, so
# generation length and the allocated window can't drift apart.
DEFAULT_MAX_NEW_TOKENS = GeneratedTextQA.max_new_tokens

SCORE_COL_NAME = "risk_score"
LABEL_COL_NAME = "label"


@dataclass(frozen=True)
class InferenceConfig:
    """Typed container for a classifier's inference/generation knobs.

    Replaces the old untyped ``inference_kwargs`` dict + loose ``temperature`` /
    ``seed`` params: one frozen, self-documenting object read via classifier
    properties instead of stringly-typed dict lookups scattered across backends.

    All fields except ``batch_size`` affect the *output* and therefore result
    identity (see :meth:`identity_dict`): ``context_size``/``max_new_tokens``
    truncate or cap generation, ``reasoning`` toggles thinking, ``temperature``
    and ``seed`` govern sampling. ``batch_size`` is a pure throughput knob and
    must NOT change results, so it is excluded from the classifier hash.
    """

    context_size: int = DEFAULT_CONTEXT_SIZE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    reasoning: str | None = None
    temperature: float | None = None
    seed: int = DEFAULT_SEED
    # Throughput-only; excluded from result identity.
    batch_size: int = field(default=DEFAULT_BATCH_SIZE)

    def identity_dict(self) -> dict:
        """Result-affecting fields (everything but the throughput-only batch_size)."""
        params = dataclasses.asdict(self)
        params.pop("batch_size")
        return params


class LLMClassifier(BaseEstimator, ClassifierMixin, ABC):
    """An interface to produce risk scores and class predictions with an LLM."""

    # Generated-text answer-extraction failure-rate observability (shared by all
    # backends that parse answers from generated text). Warn once the rate crosses
    # the threshold (after ≥ MIN_SAMPLES) and again at each 200-sample boundary, so
    # a run with mostly-failing extractions surfaces instead of silently degrading.
    _REGEX_FAILURE_WARN_THRESHOLD = 0.25
    _REGEX_FAILURE_WARN_MIN_SAMPLES = 20

    def __init__(
        self,
        model_name: str,
        task: TaskMetadata | str,
        encode_row: EncodeRowCallable | None = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        prompt_config: PromptConfig | None = None,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier object.

        Parameters
        ----------
        model_name : str
            The model name or ID.
        task : TaskMetadata | str
            The task metadata object or name of an already created task.
        encode_row : Callable[[pd.Series], str], optional
            The function used to encode tabular rows into natural text. If not
            provided, will use the default encoding function for the task.
        threshold : float, optional
            The classification threshold to use when outputting binary
            predictions, by default 0.5. Must be between 0 and 1. Will be
            re-calibrated if `fit` is called.
        correct_order_bias : bool, optional
            Whether to correct ordering bias in multiple-choice Q&A questions,
            by default True.
        **inference_kwargs
            The inference/generation knobs collected into an
            :class:`InferenceConfig`: ``context_size``, ``max_new_tokens``,
            ``reasoning``, ``temperature``, ``seed``, ``batch_size``. See that
            class for semantics; ``temperature`` is resolved per question via
            :meth:`_resolve_temperature`.
        """

        # Set classifier metadata
        self._model_name = model_name
        self._task = TaskMetadata.get_task(task) if isinstance(task, str) else task
        self._prompt_config = prompt_config or PromptConfig.from_dict(pv={}, task=self.task)

        self._threshold = threshold
        self._threshold_fitted_on = 0
        self._threshold_obj = "balanced_accuracy"  ##TODO: remove (but will change benchmark hash)
        self._correct_order_bias = correct_order_bias

        # Collect the inference/generation knobs into one typed config. Reject
        # unknown kwargs instead of silently swallowing them: prompt-shaping args
        # removed in the refactor (e.g. `custom_prompt_prefix`) would otherwise
        # land here unused and silently change behavior.
        valid_keys = {f.name for f in dataclasses.fields(InferenceConfig)}
        unknown = set(inference_kwargs) - valid_keys
        if unknown:
            raise TypeError(
                f"Unexpected keyword argument(s) {sorted(unknown)}. Valid inference kwargs "
                f"are {sorted(valid_keys)}; prompt-shaping options removed in the refactor "
                f"(e.g. 'custom_prompt_prefix') are now set via `prompt_config` or the CLI "
                f"`--variation` flag."
            )
        self._inference = InferenceConfig(**inference_kwargs)

        # Choose the encode function: user-supplied one wins; otherwise
        # `_default_encode_row` picks the default. The base returns a plain prompt
        # (tokenizer-agnostic); local backends override it to route generated-text
        # based tasks through the chat template using their own tokenizer.
        self._encode_row: EncodeRowCallable = encode_row or self._default_encode_row()

        # Fixed sklearn parameters
        self.classes_ = np.array([0, 1])
        self._is_fitted = False

        # Track answer extraction failures across batches so we can warn
        # if a non-trivial fraction of samples fall back to the silent 0.5
        # default — otherwise a benchmark with collapsed AUC looks "successful".
        # Shared across backends.
        self._regex_total = 0
        self._regex_failed = 0

        # Generated-text debug logging (opt-in via env vars): log the raw
        # prompt/generation/score for the first N generations, or all of them.
        # Shared across backends.
        self._log_generations_all = getenv("FOLKTEXTS_LOG_GENERATIONS", "0").strip() in {"1", "true", "True"}
        try:
            self._log_generations_first_n = int(getenv("FOLKTEXTS_LOG_GENERATIONS_FIRST_N", "3"))
        except ValueError:
            self._log_generations_first_n = 3
        self._logged_generations_count = 0

    def _check_reasoning_compatible_with_decoding(self) -> None:
        """Reject reasoning/thinking combined with token-probability decoding.

        Mirrors `Benchmark._validate_config`: thinking emits a trace *before* the
        answer, so the multi-pass logprob reader (1-2 forward passes) never sees
        the answer token, and Claude/OpenAI don't return logprobs with thinking
        enabled. Requires the generated-text path instead.
        """
        reasoning = self._inference.reasoning
        if reasoning is not None and str(reasoning) != "0" and not self.task.question.use_generated_text:
            raise ValueError(
                f"Reasoning/thinking (`reasoning={reasoning!r}`) requires text-based "
                "answer extraction (a generated-text question type / "
                "`use_generated_text=True`): the logprob path reads untempered "
                "top-logprobs over 1-2 forward passes and cannot capture an answer "
                "that follows a thinking trace (and Claude/OpenAI don't return "
                "logprobs with thinking)."
            )

    def __hash__(self) -> int:
        """Generate a unique hash for this object."""

        # All parameters that affect the model's behavior. The inference config's
        # result-affecting fields (context_size, max_new_tokens, reasoning,
        # temperature, seed — but NOT the throughput-only batch_size) enter here
        # so runs that differ only in e.g. `reasoning` don't collide in the cache.
        hash_params = dict(
            model_name=self.model_name,
            task_hash=hash(self.task),
            prompt_config_hash=hash(self.prompt_config),
            correct_order_bias=self.correct_order_bias,
            threshold=self.threshold,
            inference_config=self._inference.identity_dict(),
            encode_row_hash=hash_function(self.encode_row),
        )

        return int(hash_dict(hash_params), 16)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def task(self) -> TaskMetadata:
        return self._task

    @property
    def prompt_config(self) -> PromptConfig:
        """The :class:`~folktexts.prompting.PromptConfig` used to render this classifier's prompts."""
        return self._prompt_config

    @property
    def encode_row(self) -> EncodeRowCallable:
        return self._encode_row

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if not 0 <= value <= 1:
            logging.error(f"Threshold must be between 0 and 1; got {value}.")

        # Clip threshold to valid range
        self._threshold = np.clip(value, 0, 1)
        logging.warning(f"Setting {self.model_name} threshold to {self._threshold}.")

    @property
    def correct_order_bias(self) -> bool:
        return self._correct_order_bias

    @correct_order_bias.setter
    def correct_order_bias(self, value: bool):
        self._correct_order_bias = value
        logging.warning(f"Setting {self.model_name} correct_order_bias to {value}.")

    @property
    def inference_config(self) -> InferenceConfig:
        """The typed :class:`InferenceConfig` holding this classifier's generation knobs."""
        return self._inference

    @property
    def seed(self) -> int:
        return self._inference.seed

    @property
    def context_size(self) -> int:
        return self._inference.context_size

    @property
    def batch_size(self) -> int:
        return self._inference.batch_size

    @property
    def reasoning(self) -> str | None:
        """Reasoning/thinking effort (tri-state): ``None`` = no reasoning control,
        ``'0'`` = thinking off, any other value = thinking on."""
        return self._inference.reasoning

    @property
    def temperature(self) -> float | None:
        """The explicit sampling-temperature override, or ``None`` to defer to
        each question type's :attr:`~folktexts.qa_interface.QAInterface.default_temperature`."""
        return self._inference.temperature

    @property
    def max_new_tokens(self) -> int:
        """Output-token budget for generated-text prompting, shared by every backend.

        Single resolution point (mirroring :meth:`_resolve_temperature`) so the
        generation length is read from one place. Defaults to
        ``GeneratedTextQA.max_new_tokens``, the same value ``run_benchmark`` uses
        to size vLLM's ``max_model_len``, so generation and the window can't drift.
        """
        return self._inference.max_new_tokens

    def _resolve_temperature(
        self,
        question: MultipleChoiceQA | DirectNumericQA,
    ) -> float:
        """Return the sampling temperature to use when generating text for ``question``.

        Resolution order:
        1. An explicit classifier-level override (`self.temperature`) always wins.
        2. Otherwise, when reasoning is active, force ``1.0`` — greedy
           decoding is unreliable for (or outright rejected by) thinking models.
        3. Otherwise fall back to the question type's ``default_temperature``
           (``0.0`` greedy — plain generated-text stays deterministic).

        Only the text-generation paths call this (multiple-choice / direct-numeric
        read untempered token probabilities on every backend and never sample).
        """
        # TODO (per-family thinking temperature): blanket 1.0 is *required* by Claude
        # with thinking but only acceptable (~0.6 is better) for Qwen3/DeepSeek.
        if self._inference.temperature is not None:
            return self._inference.temperature
        if self._inference.reasoning is not None:
            return 1.0
        return question.default_temperature

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted

    @staticmethod
    def _get_positive_class_scores(risk_scores: np.ndarray) -> np.ndarray:
        """Helper function to get positive class scores from risk scores."""
        if len(risk_scores.shape) > 1:
            return risk_scores[:, -1]
        else:
            return risk_scores

    @staticmethod
    def _make_predictions_multiclass(pos_class_scores: np.ndarray) -> np.ndarray:
        """Converts positive class scores to multiclass scores."""
        return np.column_stack([1 - pos_class_scores, pos_class_scores])

    @staticmethod
    def _apply_nan_policy(scores: np.ndarray, impute_failed_as_uniform: bool) -> tuple[np.ndarray, np.ndarray, int]:
        """Resolve NaN (failed / non-parsable) risk scores per the chosen policy.

        Returns ``(scores, keep_mask, n_failed)``. When
        ``impute_failed_as_uniform`` is True, NaNs are replaced by 0.5 (the
        uniform / max-entropy prior) and ``keep_mask`` is all-True (nothing is
        dropped); otherwise the scores are returned unchanged and ``keep_mask``
        marks the non-NaN rows for the caller to drop. ``n_failed`` is the raw
        NaN count regardless of policy.
        """
        n_failed = int(np.isnan(scores).sum())
        if impute_failed_as_uniform and n_failed:
            logging.warning(f"Imputing {n_failed} failed/non-parsable score(s) with 0.5 (uniform prior).")
            scores = np.where(np.isnan(scores), 0.5, scores)
        keep_mask = ~np.isnan(scores)
        return scores, keep_mask, n_failed

    def _maybe_warn_regex_extraction_failure_rate(self) -> None:
        """Warn if the generated-text answer-extraction failure rate is high.

        Fires once the rate crosses ``_REGEX_FAILURE_WARN_THRESHOLD`` (after at
        least ``_REGEX_FAILURE_WARN_MIN_SAMPLES``) and again at each 200-sample
        boundary. Callers on the generated-text path bump ``_regex_total`` per
        response and ``_regex_failed`` when the parsed estimate is NaN.
        """
        if self._regex_total < self._REGEX_FAILURE_WARN_MIN_SAMPLES:
            return
        if self._regex_total % 200 != 0 and self._regex_total != self._REGEX_FAILURE_WARN_MIN_SAMPLES:
            return
        rate = self._regex_failed / self._regex_total
        if rate >= self._REGEX_FAILURE_WARN_THRESHOLD:
            logging.warning(
                f"Answer extraction failed for {self._regex_failed}/{self._regex_total} "
                f"generated responses ({rate:.1%}); failed passes fall back to 0.5 and rows "
                f"with no valid pass become NaN. Inspect generations via "
                f"FOLKTEXTS_LOG_GENERATIONS_FIRST_N (local backends)."
            )

    def _should_log_generation(self) -> bool:
        """Whether to log the next raw prompt/generation pair (debug aid).

        Opt-in via ``FOLKTEXTS_LOG_GENERATIONS`` (all) or bounded to the first
        ``FOLKTEXTS_LOG_GENERATIONS_FIRST_N`` generations. Callers increment
        ``_logged_generations_count`` after emitting the log.
        """
        if self._log_generations_all:
            return True
        return self._logged_generations_count < max(self._log_generations_first_n, 0)

    def _default_encode_row(self) -> EncodeRowCallable:
        """Build the default encode function when the user supplies none.

        The base is tokenizer-agnostic and returns a plain prompt. Local backends
        (which own a tokenizer) override this to route generated-text tasks
        through the chat template.
        """
        return partial(default_encode_row_prompt, task=self.task, prompt_config=self._prompt_config)

    def _maybe_log_generation(self, prompt: str, generated_text: str, risk_estimate: float) -> None:
        """Log one raw prompt/generation/score triple for debugging, if enabled.

        No-op unless `_should_log_generation()`; increments the logged counter
        when it emits. Centralizes the format shared by all generated-text backends.
        """
        if not self._should_log_generation():
            return
        sep = "\n" + "-" * 60 + "\n"
        sep_head = "\n" + "=" * 60 + "\n"
        logging.info(
            f"{sep_head}[Sample {self._logged_generations_count + 1}]\n"
            f"PROMPT:\n{prompt}{sep}"
            f"GENERATED ANSWER:\n{generated_text}{sep}"
            f"EXTRACTED RISK SCORE: {risk_estimate:.6f}{sep_head}"
        )
        self._logged_generations_count += 1

    def _maybe_log_logprobs(
        self,
        prompt: str,
        per_pass_token_probs: list[dict[str, float]],
        risk_estimate: float,
        *,
        top_k: int = 10,
    ) -> None:
        """Log one prompt + answer-token probability distribution + score, if enabled.

        Counterpart to `_maybe_log_generation` for the token-probability decode
        path, which has no generated text — the answer is read from the answer
        tokens' probabilities. `per_pass_token_probs` is one dict per forward pass
        mapping a token string to its probability (already decoded by the backend,
        since token-id→string differs per backend). Gated by the same
        `_should_log_generation()` toggle and shares its counter. Callers should
        build `per_pass_token_probs` only when logging is enabled (guard with
        `_should_log_generation()`) — decoding the top-K per sample is otherwise
        wasted work.
        """
        if not self._should_log_generation():
            return
        sep = "\n" + "-" * 60 + "\n"
        sep_head = "\n" + "=" * 60 + "\n"
        pass_lines = []
        for pass_idx, token_probs in enumerate(per_pass_token_probs):
            top = sorted(token_probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            rendered = ", ".join(f"{tok!r}={prob:.4f}" for tok, prob in top)
            pass_lines.append(f"  pass {pass_idx}: {rendered}")
        dist = "\n".join(pass_lines) if pass_lines else "  (none)"
        logging.info(
            f"{sep_head}[Sample {self._logged_generations_count + 1}]\n"
            f"PROMPT:\n{prompt}{sep}"
            f"ANSWER-TOKEN PROBABILITIES (top-{top_k} per pass):\n{dist}{sep}"
            f"EXTRACTED RISK SCORE: {risk_estimate:.6f}{sep_head}"
        )
        self._logged_generations_count += 1

    def fit(
        self,
        X,
        y,
        *,
        false_pos_cost=1.0,
        false_neg_cost=1.0,
        threshold_obj="balanced_accuracy",
        impute_failed_as_uniform=False,
        **kwargs,
    ):
        """Uses the provided data sample to fit the prediction threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data to run inference on.
        y : pd.Series
            True binary labels for the data.
        false_pos_cost : float, optional
            Cost of a false positive; used to weight the threshold search.
        false_neg_cost : float, optional
            Cost of a false negative; used to weight the threshold search.
        threshold_obj : str, optional
            Metric to maximise when searching for the best threshold
            (e.g. ``"balanced_accuracy"``).
        impute_failed_as_uniform : bool, optional
            If True, failed (NaN) scores are set to 0.5 (uniform prior) and kept;
            if False (default), they are dropped before fitting the threshold.
        """

        # Compute risk estimates for the data
        y_pred_scores = self._get_positive_class_scores(self.predict_proba(X, **kwargs))

        # Resolve failed (NaN) scores per policy: impute with 0.5 or drop before
        # threshold fitting. Under imputation `nan_mask` is all-True (no drop).
        y_pred_scores, nan_mask, _ = self._apply_nan_policy(y_pred_scores, impute_failed_as_uniform)
        if not nan_mask.all():
            logging.warning(f"fit: dropping {(~nan_mask).sum()} NaN score(s) out of {len(y_pred_scores)}.")
            y = y[nan_mask]
            y_pred_scores = y_pred_scores[nan_mask]

        # Compute the best threshold for the given data
        self.threshold = compute_best_threshold(
            y,
            y_pred_scores,
            false_pos_cost=false_pos_cost,
            false_neg_cost=false_neg_cost,
            maximize=threshold_obj,
        )
        self._threshold_obj = threshold_obj  ## TODO: save in llm_clf._threshold_obj before calling fun

        # Update sklearn is_fitted status
        self._is_fitted = True
        return self

    def _load_predictions_from_disk(
        self,
        predictions_save_path: str | Path,
        data: pd.DataFrame,
    ) -> np.ndarray | None:
        """Attempts to load pre-computed predictions from disk."""

        # Load predictions from disk
        predictions_save_path = Path(predictions_save_path).with_suffix(".csv")
        predictions_df = pd.read_csv(predictions_save_path, index_col=0)

        # Check if index matches our current dataframe
        if predictions_df.index.equals(data.index):
            return predictions_df[SCORE_COL_NAME].values
        else:
            logging.error("Saved predictions do not match the current dataframe.")
            return None

    def predict(
        self,
        data: pd.DataFrame,
        predictions_save_path: str | Path | None = None,
        labels: pd.Series | np.ndarray = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Returns binary predictions for the given data."""
        risk_scores = self.predict_proba(
            data,
            predictions_save_path=predictions_save_path,
            labels=labels,
        )
        return (self._get_positive_class_scores(risk_scores) >= self.threshold).astype(int)

    def predict_proba(
        self,
        data: pd.DataFrame,
        predictions_save_path: str | Path | None = None,
        labels: pd.Series | np.ndarray = None,
    ) -> np.ndarray:
        """Returns probability estimates for the given data.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to compute risk estimates for.
        predictions_save_path : str | Path, optional
            If provided, will save the computed risk scores to this path in
            disk. If the path exists, will attempt to load pre-computed
            predictions from this path.
        labels : pd.Series | np.ndarray, optional
            The labels corresponding to the provided data. Not required to
            compute predictions. Will only be used to save alongside predictions
            to disk.

        Returns
        -------
        risk_scores : np.ndarray
            The risk scores for the given data.
        """
        # Fail fast on an impossible reasoning/decoding combination before any
        # disk I/O or inference. Checked here (not in __init__) because the
        # decoded question is `self.task.question`, whose mode can be reconfigured
        # after construction (mutable task flags); this reads its current value.
        # (Safety net for direct classifier use bypassing Benchmark._validate_config.)
        self._check_reasoning_compatible_with_decoding()

        # Validate arguments
        if labels is not None and predictions_save_path is None:
            logging.error(
                "** Ignoring `labels` argument as `predictions_save_path` was not provided. **"
                "The `labels` argument is only used in conjunction with "
                "`predictions_save_path` to save alongside predictions to disk. "
            )

        if predictions_save_path is not None:
            # Check if `predictions_save_path` exists and load predictions if possible
            logging.info(
                f"Check if predictions_save_path '{predictions_save_path}' exists:{Path(predictions_save_path).exists()}"
            )
            if Path(predictions_save_path).exists():
                result = self._load_predictions_from_disk(predictions_save_path, data=data)
                if result is not None:
                    logging.info(f"Loaded predictions from {predictions_save_path}.")
                    return self._make_predictions_multiclass(result)
                else:
                    logging.error(
                        f"Failed to load predictions from {predictions_save_path}. "
                        f"Re-computing predictions and overwriting local file..."
                    )

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"`data` must be a pd.DataFrame, received {type(data)} instead.")

        # Compute risk estimates
        risk_scores = self.compute_risk_estimates_for_dataframe(
            df=data, save_intermed={"path": predictions_save_path, "labels": labels}
        )

        # Save to disk if `predictions_save_path` is provided
        if predictions_save_path is not None:
            predictions_save_path = Path(predictions_save_path).with_suffix(".csv")
            logging.info(f"Saving predictions to {predictions_save_path}")

            predictions_df = pd.DataFrame(risk_scores, index=data.index, columns=[SCORE_COL_NAME])
            predictions_df[LABEL_COL_NAME] = labels
            predictions_df.to_csv(predictions_save_path, index=True, mode="w")

        return self._make_predictions_multiclass(risk_scores)

    @abstractmethod
    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> tuple[np.ndarray, list]:
        """Query model with a batch of prompts and return risk estimates."""
        raise NotImplementedError("Calling an abstract method :: Use one of the subclasses of LLMClassifier.")

    def compute_risk_estimates_for_dataframe(
        self,
        df: pd.DataFrame,
        save_intermed: dict = {"path": None, "labels": None},
    ) -> np.ndarray:
        """Compute risk estimates for a specific dataframe (internal helper function).

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to compute risk estimates for.
        save_intermed: dict
            A dictionary containing information for saving intermediate results. Should contain the keys:
                - 'path': str | Path, optional
                - 'labels': pd.Series | np.ndarray, optional
        Returns
        -------
        risk_scores : np.ndarray
            The risk estimates for each row in the dataframe.
        """

        # Initialize risk scores and other constants
        fill_value = -1
        risk_scores = np.empty(len(df))
        risk_scores.fill(fill_value)  # fill with -1's

        batch_size = self.batch_size or DEFAULT_BATCH_SIZE
        context_size = self.context_size or DEFAULT_CONTEXT_SIZE
        num_batches = math.ceil(len(df) / batch_size)

        # Get questions to ask
        q = self.task.question
        questions = [q]
        if self.correct_order_bias:
            if isinstance(q, DirectNumericQA):
                logging.info("No need to correct ordering bias for DirectNumericQA prompting.")
            elif isinstance(q, MultipleChoiceQA):
                questions = list(MultipleChoiceQA.create_answer_keys_permutations(q))
            else:
                logging.error(f"Unknown question type '{type(q)}'; cannot correct ordering bias.")

        # Prepare storage for model responses
        model_outputs = []  # either text or tlp

        # Path for saving results (loop-invariant; None disables saving). `batch_path`
        # is the transient per-checkpoint predictions file, removed after the run.
        path = Path(save_intermed["path"]) if save_intermed.get("path") is not None else None
        batch_path = None
        # Responses are appended incrementally to a single CSV; this tracks how many
        # response rows have been written so each save appends only the new ones.
        n_resp_written = 0

        # Compute risk estimates per batch
        for batch_idx in tqdm(range(num_batches), desc="Computing risk estimates"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_data = df.iloc[start_idx:end_idx]
            batch_row_ids = batch_data.index.values

            # Materialize row-Series once per batch; `iterrows()` rebuilds a
            # Series (with dtype coercion) per row, and under order-bias
            # correction the inner question loop iterates N_permutations times.
            batch_rows = [row for _, row in batch_data.iterrows()]

            batch_risk_scores = np.empty((len(batch_data), len(questions)))
            for q_idx, q in enumerate(questions):
                # Encode batch data into natural text prompts
                # TODO: potential improvement: encode outside loop with question placeholder, only replace placeholder
                data_texts_batch = [self.encode_row(row, question=q) for row in batch_rows]

                # Query the model with the batch of data
                risk_estimates_batch, responses_batch = self._query_prompt_risk_estimates_batch(
                    prompts_batch=data_texts_batch,
                    question=q,
                    context_size=context_size,
                )

                # Store risk estimates for current question
                batch_risk_scores[:, q_idx] = np.clip(risk_estimates_batch, 0, 1)
                # Record generated-text responses for text-generation path
                if q.use_generated_text:
                    is_mcq = isinstance(q, MultipleChoiceQA)
                    for i, resp in enumerate(responses_batch):
                        resp = resp or {}
                        response_text = resp.get("response")
                        if response_text is None:
                            extracted_answer = ""
                        elif is_mcq:
                            # q is a MultipleChoiceQA here: map the parsed value to its answer key.
                            extracted_answer = q.get_answer_key_from_value(  # type: ignore[union-attr]
                                q.get_answer_from_generated_text(response_text)
                            )
                        else:
                            # DirectNumericQA text path: the extracted answer is the
                            # parsed probability (already computed as the risk estimate).
                            extracted_answer = risk_estimates_batch[i]
                        # Normalize failures (None / NaN) to an empty cell.
                        if extracted_answer is None or (isinstance(extracted_answer, float) and np.isnan(extracted_answer)):
                            extracted_answer = ""
                        model_outputs.append(
                            {
                                "row_idx": batch_row_ids[i],
                                "question_idx": q_idx,
                                "prompt": data_texts_batch[i],  # includes row information and question
                                "reasoning": resp.get("reasoning", ""),
                                "response": resp.get("response", ""),
                                "extracted_answer": extracted_answer,
                                "reasoning_tokens": resp.get("reasoning_tokens"),
                            }
                        )

            # Average the order-bias permutations into the final risk score.
            # A failed pass (NaN, e.g. unparsable generated text) hedges to 0.5 rather than being dropped:
            # dropping would keep the lone surviving hard label and overstate confidence (e.g. [1.0, NaN]
            # resolves to 0.75, not 1.0). Rows where every pass failed stay NaN for the eval-time
            # `impute_failed_as_uniform` policy.
            # Only applies to text-generation path, no-op for token-prob (never NaN).
            all_failed = np.isnan(batch_risk_scores).all(axis=1)
            hedged = np.where(np.isnan(batch_risk_scores), 0.5, batch_risk_scores)
            row_scores = hedged.mean(axis=1)
            row_scores[all_failed] = np.nan
            risk_scores[start_idx:end_idx] = row_scores

            # log ambiguous items: multiple answers, not extractable or permutation of Q lead to different outcomes
            tol = 1e-8
            arr = batch_risk_scores.reshape(-1, 1) if batch_risk_scores.ndim == 1 else batch_risk_scores
            undecided_or_disagree = ~(np.isclose(arr, 0, atol=tol) | np.isclose(arr, 1, atol=tol))
            if np.any(undecided_or_disagree):
                idx_unclear = np.nonzero(undecided_or_disagree)[0]  # only row indices (1D)
                msg = (
                    f"Risk scores: {batch_risk_scores[idx_unclear]}"
                    f"\nRisk scores mean: {risk_scores[start_idx:end_idx][idx_unclear]}"
                )
                if questions[0].use_generated_text:
                    tmp_texts = [
                        f"response {i}\n{item}\n\n"
                        for i, item in enumerate(model_outputs[-len(batch_data) * len(questions) :])
                        if i in idx_unclear
                    ]
                    logging.debug(msg + f"\nCorresponding responses: {tmp_texts}")
                else:
                    logging.debug(msg)

            # Crash-safety checkpoint every 10 batches: rewrite the transient
            # predictions file, and append the responses computed since the last one.
            if batch_idx % 10 == 0 and path is not None:
                batch_path = path.with_stem(path.stem + "_batch")
                batch_labels = save_intermed["labels"]
                sliced_labels = batch_labels[:end_idx] if batch_labels is not None else None
                self._save_predictions(batch_path, risk_scores[:end_idx], sliced_labels, df.index[:end_idx])
                if questions[0].use_generated_text:
                    n_resp_written = self._append_responses(
                        self._responses_path(path),
                        model_outputs,
                        start=n_resp_written,
                        risk_scores=risk_scores[:end_idx],
                        labels=sliced_labels,
                        row_index=df.index[:end_idx],
                    )

        # Check that all risk scores were computed
        assert not np.isclose(risk_scores, fill_value).any()

        # Final responses append: the last partial batches after the last checkpoint
        # (predictions are finalized separately by predict_proba, the sklearn layer).
        if path is not None and questions[0].use_generated_text:
            self._append_responses(
                self._responses_path(path),
                model_outputs,
                start=n_resp_written,
                risk_scores=risk_scores,
                labels=save_intermed["labels"],
                row_index=df.index,
            )
        # Drop the transient predictions checkpoint once the full run completed.
        if batch_path is not None and Path(batch_path).exists() and str(batch_path).endswith("_batch.csv"):
            logging.info(f"Removing file '{batch_path}'.")
            remove(batch_path)
        return risk_scores

    @staticmethod
    def _responses_path(predictions_path: Path) -> Path:
        """Sibling responses-CSV path for a predictions path (`…predictions…` -> `…responses…`)."""
        return predictions_path.with_stem(predictions_path.stem.replace("_predictions", "_responses"))

    def _save_predictions(
        self,
        path: Path,
        risk_scores: np.ndarray,
        labels: pd.Series | np.ndarray | None,
        row_index: pd.Index,
    ) -> None:
        """Write the transient per-checkpoint predictions CSV (`*_batch.csv`), keyed
        by the true row index. `labels` aligns on index for a Series and positionally
        for an ndarray; `None` yields a null label column. Removed once the run
        completes (the final predictions come from ``predict_proba``)."""
        predictions_df = pd.DataFrame(risk_scores, index=row_index, columns=[SCORE_COL_NAME])
        predictions_df[LABEL_COL_NAME] = labels
        logging.info(f"Saving predictions checkpoint to {path}")
        predictions_df.to_csv(path, index=True, mode="w")

    def _append_responses(
        self,
        path: Path,
        responses: list,
        *,
        start: int,
        risk_scores: np.ndarray,
        labels: pd.Series | np.ndarray | None,
        row_index: pd.Index,
    ) -> int:
        """Append `responses[start:]` to the persistent responses CSV, annotating each
        row with its final score/label (matched by row index — a row's score is final
        once its batch is done, so appended rows never need rewriting).

        The first write of the run (``start == 0``) truncates any stale file and emits
        the header; later writes append — so a re-run never accumulates onto old data
        and each checkpoint only pays for the *new* rows. Returns the new written count.
        """
        new = responses[start:]
        if not new:
            return start
        scores = pd.Series(np.asarray(risk_scores), index=row_index)
        response_df = pd.DataFrame(new)
        response_df[SCORE_COL_NAME] = response_df["row_idx"].map(scores)
        response_df[LABEL_COL_NAME] = (
            response_df["row_idx"].map(pd.Series(np.asarray(labels), index=row_index)) if labels is not None else None
        )
        first = start == 0
        logging.info(f"{'Saving' if first else 'Appending'} responses ({len(new)} rows) to {path}")
        response_df.to_csv(path, index=False, encoding="utf-8", mode="w" if first else "a", header=first)
        return len(responses)

    def compute_risk_estimates_for_dataset(
        self,
        dataset: Dataset,
    ) -> dict[str, np.ndarray]:
        """Computes risk estimates for each row in the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to compute risk estimates for.

        Returns
        -------
        results : dict[str, np.ndarray]
            The risk estimates for each data type in the dataset (usually "train",
            "val", "test").
        """
        data_types = {
            "train": dataset.get_train()[0],
            "test": dataset.get_test()[0],
        }
        if dataset.get_val() is not None:
            data_types["val"] = dataset.get_val()[0]

        results = {
            data_type: self.compute_risk_estimates_for_dataframe(
                df=df,
            )
            for data_type, df in data_types.items()
        }

        return results
