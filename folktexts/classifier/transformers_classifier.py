"""Module for using huggingface transformers models as classifiers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from folktexts.llm_utils import generate_text_batch, query_model_batch_multiple_passes
from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata
from folktexts.token_tracker import TokenTracker

from .._utils import hash_dict
from .base import EncodeRowCallable, LLMClassifier


class TransformersLLMClassifier(LLMClassifier):
    """Use a huggingface transformers model to produce risk scores."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task: TaskMetadata | str,
        encode_row: EncodeRowCallable = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        seed: int = 42,
        token_tracker: TokenTracker | None = None,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier based on a huggingface transformers model.

        Parameters
        ----------
        model : AutoModelForCausalLM
            The torch language model to use for inference.
        tokenizer : AutoTokenizer
            The tokenizer used to train the model.
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
        seed : int, optional
            The random seed - used for reproducibility.
        token_tracker : TokenTracker, optional
            A :class:`~folktexts.token_tracker.TokenTracker` instance for
            recording token usage. If *None*, no tracking is performed.
        **inference_kwargs
            Additional keyword arguments to be used at inference time. Options
            include `context_size` and `batch_size`.
        """
        # Transformers objects for the model and tokenizer
        self._model = model
        self._tokenizer = tokenizer
        self.token_tracker = token_tracker

        # Fetch name for transformers model
        model_name = Path(self._model.name_or_path).name

        super().__init__(
            model_name=model_name,
            task=task,
            encode_row=encode_row,
            correct_order_bias=correct_order_bias,
            threshold=threshold,
            seed=seed,
            **inference_kwargs,
        )

        # Logging controls (used mainly for text generation debugging).
        # By default, log only the first N prompt/generation pairs; users can
        # enable logging all generations via env var or CLI wrapper.
        self._log_generations_all = os.getenv("FOLKTEXTS_LOG_GENERATIONS", "0").strip() in {"1", "true", "True"}
        try:
            self._log_generations_first_n = int(os.getenv("FOLKTEXTS_LOG_GENERATIONS_FIRST_N", "3"))
        except ValueError:
            self._log_generations_first_n = 3
        self._logged_generations_count = 0

        # Track answer extraction failures across batches so we can warn
        # if a non-trivial fraction of samples fall back to the silent 0.5
        # default — otherwise a benchmark with collapsed AUC looks "successful".
        self._regex_total = 0
        self._regex_failed = 0

    def _should_log_generation(self) -> bool:
        """Return True if we should log the next prompt/generation pair."""
        if self._log_generations_all:
            return True
        return self._logged_generations_count < max(self._log_generations_first_n, 0)

    # Warn the first time the failure rate crosses 25% (after ≥20 samples) and
    # again at every 200-sample boundary, so a benchmark with mostly-failing
    # extractions surfaces in the logs instead of silently collapsing AUC to 0.5.
    _REGEX_FAILURE_WARN_THRESHOLD = 0.25
    _REGEX_FAILURE_WARN_MIN_SAMPLES = 20

    def _maybe_warn_regex_extraction_failure_rate(self) -> None:
        if self._regex_total < self._REGEX_FAILURE_WARN_MIN_SAMPLES:
            return
        if self._regex_total % 200 != 0 and self._regex_total != self._REGEX_FAILURE_WARN_MIN_SAMPLES:
            return
        rate = self._regex_failed / self._regex_total
        if rate >= self._REGEX_FAILURE_WARN_THRESHOLD:
            logging.warning(
                f"Probability extraction failed via regex for "
                f"{self._regex_failed}/{self._regex_total} samples "
                f"({rate:.1%}); these fall back to 0.5 and will collapse AUC. "
                f"Inspect generations with FOLKTEXTS_LOG_GENERATIONS_FIRST_N."
            )

    def __hash__(self) -> int:
        """Generate a unique hash for the LLMClassifier object."""

        # All parameters that affect the model's behavior
        hash_params = dict(
            super_hash=super().__hash__(),
            model_size=self._model.num_parameters(),
            tokenizer_vocab_size=self._tokenizer.vocab_size,
        )

        return int(hash_dict(hash_params), 16)

    @property
    def model(self) -> AutoModelForCausalLM:
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> tuple[np.ndarray, list]:
        """Query model with a batch of prompts and return risk estimates.

        Parameters
        ----------
        prompts_batch : list[str]
            A batch of string prompts to query the model with.
        question : MultipleChoiceQA | DirectNumericQA
            The question (`QAInterface`) object to use for querying the model.
        context_size : int, optional
            The maximum context size to consider for each input (in tokens).

        Returns
        -------
        risk_estimates : np.ndarray
            The risk estimates for each prompt in the batch.
        """
        # Count prompt tokens once (shared by both branches)
        if self.token_tracker is not None:
            prompt_tokens = sum(len(self._tokenizer.encode(p, add_special_tokens=False)) for p in prompts_batch)

        # Handle ChainOfThoughtQA with text generation
        #  (kept for future use: numeric questions with use_generated_text can extract
        # the probability from the generated text via regex instead of token log-probs)
        # if isinstance(question, ChainOfThoughtQA):
        #     # Pass enable_thinking to generate_text_batch:
        #     # - True: enable thinking mode (uses chat template with enable_thinking=True)
        #     # - False: explicitly disable thinking mode (uses chat template with enable_thinking=False)
        #     # Always apply chat template for ChainOfThoughtQA to properly format the prompt
        #     generated_texts = generate_text_batch(
        #         text_inputs=prompts_batch,
        #         model=self.model,
        #         tokenizer=self.tokenizer,
        #         max_new_tokens=question.max_new_tokens,
        #         context_size=context_size or self.inference_kwargs["context_size"],
        #         enable_thinking=question.enable_thinking,
        #         system_prompt=(self.prompt_config.system_prompt() if self.prompt_config.system_prompt is not None else None),
        #         temperature=self._resolve_temperature(question),
        #         seed=self.seed,
        #     )

        #     # Extract probability from generated text and log each sample
        #     risk_estimates_batch = []
        #     for idx, (prompt, generated_text) in enumerate(zip(prompts_batch, generated_texts)):
        #         extracted = question.extract_probability_from_text(generated_text)
        #         self._regex_total += 1
        #         if extracted is None:
        #             self._regex_failed += 1
        #         risk_estimate = 0.5 if extracted is None else extracted
        #         risk_estimates_batch.append(risk_estimate)
        #         self._maybe_warn_regex_extraction_failure_rate()

        #         if self._should_log_generation():
        #             # Log prompt, generated answer, and extracted risk score at INFO level
        #             logging.info(
        #                 ("\n" + "=" * 60 + "\n")
        #                 + f"[ChainOfThoughtQA Sample {self._logged_generations_count + 1}]"
        #                 + ("\n" + "=" * 60 + "\n")
        #                 + "PROMPT:\n"
        #                 + prompt
        #                 + ("\n" + "-" * 60 + "\n")
        #                 + "GENERATED ANSWER:\n"
        #                 + generated_text
        #                 + ("\n" + "-" * 60 + "\n")
        #                 + f"EXTRACTED RISK SCORE: {risk_estimate:.6f}\n"
        #                 + "=" * 60
        #             )
        #             self._logged_generations_count += 1

        #     return np.asarray(risk_estimates_batch, dtype=float)

        if question.use_generated_text:
            try:
                # try to apply chat
                # Query model
                # Use the system prompt from PromptConfig if available (may be None
                # to explicitly disable the role, e.g. for Gemma-style templates);
                # fall back to the QA subclass ClassVar default otherwise.
                if self.prompt_config is not None:
                    system_prompt = (
                        self.prompt_config.system_prompt() if self.prompt_config.system_prompt is not None else None
                    )
                else:
                    system_prompt = question.get_default_system_prompt()

                logging.debug(f"System prompt: {system_prompt}")

                generated_text_batch = generate_text_batch(
                    text_inputs=prompts_batch,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    context_size=context_size or self.inference_kwargs["context_size"],
                    max_new_tokens=self.inference_kwargs[
                        "max_new_tokens"
                    ],  # TODO: get max nex tokens from task or question or model?
                    reasoning=self.inference_kwargs.get("reasoning"),
                    thinking_end_token_id=None,
                    system_prompt=system_prompt,
                )

                risk_estimates_batch = [
                    question.get_answer_from_model_output(
                        text=text.get("response", ""),
                    )
                    for text in generated_text_batch
                ]

                if self.token_tracker is not None:
                    completion_tokens = sum(
                        len(self._tokenizer.encode(t.get("response", ""), add_special_tokens=False))
                        for t in generated_text_batch
                    )
                    self.token_tracker.record_batch(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        batch_size=len(prompts_batch),
                    )

                # sanitized_texts = [text.replace(";", "") for text in generated_text_batch]
                # np.assarray coerces None → nan
                # TODO; check, if wanted
                return np.asarray(risk_estimates_batch, dtype=float), generated_text_batch
            except Exception as error:
                logging.error(f"Error occurred while querying model: {error}")
                raise

        else:
            # TODO: Add support for any unicode character used as a prefix to " A".

            # Query model
            last_token_probs_batch = query_model_batch_multiple_passes(
                text_inputs=prompts_batch,
                model=self.model,
                tokenizer=self.tokenizer,
                context_size=context_size or self.inference_kwargs["context_size"],
                n_passes=question.num_forward_passes,
                digits_only=True if isinstance(question, DirectNumericQA) else False,
            )

            # Decode model output
            risk_estimates_batch = [
                question.get_answer_from_model_output(
                    ltp,
                    tokenizer_vocab=self._tokenizer.vocab,
                )
                for ltp in last_token_probs_batch
            ]

            if self.token_tracker is not None:
                # Each forward pass generates exactly one token per prompt
                completion_tokens = len(prompts_batch) * question.num_forward_passes
                self.token_tracker.record_batch(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    batch_size=len(prompts_batch),
                )

            return np.asarray(risk_estimates_batch, dtype=float), last_token_probs_batch  # type: ignore[return-value]  # ltp not used
