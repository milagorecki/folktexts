"""Module for using huggingface transformers models as classifiers."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from folktexts.llm_utils import (
    _postprocess_generated_text,
    generate_text_batch,
    get_thinking_end_token_id,
    query_model_batch_multiple_passes,
    reasoning_to_enable_thinking,
)
from folktexts.prompting import (
    chat_template_assistant_marker,
    encode_row_prompt_chat,
    tokenizer_supports_system_prompt,
    tokenizer_supports_thinking,
)
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

        self._probe_tokenizer_capabilities(self._tokenizer)

    def _default_encode_row(self) -> EncodeRowCallable:
        """Route generated-text tasks on a chat model through the chat template
        (applying enable_thinking + system prompt there); plain prompt otherwise.

        Local override of the tokenizer-agnostic base default, so a bare
        classifier (no benchmark) still templates the generation path.
        """
        if getattr(self._tokenizer, "chat_template", None) is not None and self.task.question.use_generated_text:
            logging.info("Routing default encode_row through the chat template (generated-text path).")
            return partial(
                encode_row_prompt_chat,
                task=self.task,
                tokenizer=self._tokenizer,
                prompt_config=self._prompt_config,
                enable_thinking=reasoning_to_enable_thinking(self.reasoning),
            )
        return super()._default_encode_row()

    def _probe_tokenizer_capabilities(self, tokenizer) -> None:
        """Detect and log the tokenizer's chat-template capabilities (once, diagnostic)."""
        self._has_chat_template = getattr(tokenizer, "chat_template", None) is not None
        self._supports_system_role = tokenizer_supports_system_prompt(tokenizer) if self._has_chat_template else False
        self._supports_thinking = tokenizer_supports_thinking(tokenizer) if self._has_chat_template else False
        logging.info(
            f"Tokenizer capabilities for '{self.model_name}': chat_template={self._has_chat_template}, "
            f"system_role={self._supports_system_role}, thinking={self._supports_thinking}."
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

        if question.use_generated_text:
            return self._risk_estimates_from_text(
                formatted_prompts_batch=prompts_batch,
                question=question,
                context_size=context_size,
                prompt_tokens=prompt_tokens if self.token_tracker is not None else None,
            )

        else:
            # TODO: Add support for any unicode character used as a prefix to " A".

            # Query model
            last_token_probs_batch = query_model_batch_multiple_passes(
                text_inputs=prompts_batch,
                model=self.model,
                tokenizer=self.tokenizer,
                context_size=context_size or self.context_size,
                n_passes=question.num_forward_passes,
                digits_only=True if isinstance(question, DirectNumericQA) else False,
            )

            # Decode model output
            risk_estimates_batch = []
            for prompt, ltp in zip(prompts_batch, last_token_probs_batch):
                risk_estimate = question.get_answer_from_model_output(
                    ltp,
                    tokenizer_vocab=self._tokenizer.vocab,
                )
                risk_estimates_batch.append(risk_estimate)
                if self._should_log_generation():
                    # `ltp` is the full per-pass probability distribution
                    # (n_passes, vocab_dim); take the top-K tokens per pass for
                    # the logprob-path debug log (only when logging is enabled).
                    # Invert the same `.vocab` the decode above uses (id -> token).
                    id_to_token = {tid: tok for tok, tid in self._tokenizer.vocab.items()}
                    self._maybe_log_logprobs(
                        prompt,
                        [
                            {
                                id_to_token.get(int(idx), str(int(idx))): float(pass_probs[idx])
                                for idx in np.argsort(pass_probs)[::-1][:10]
                            }
                            for pass_probs in np.asarray(ltp)
                        ],
                        risk_estimate,
                    )

            if self.token_tracker is not None:
                # Each forward pass generates exactly one token per prompt
                completion_tokens = len(prompts_batch) * question.num_forward_passes
                self.token_tracker.record_batch(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    batch_size=len(prompts_batch),
                )

            return np.asarray(risk_estimates_batch, dtype=float), last_token_probs_batch  # type: ignore[return-value]  # ltp not used

    def _risk_estimates_from_text(
        self,
        formatted_prompts_batch: list[str],
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int | None,
        prompt_tokens: int | None = None,
    ) -> tuple[np.ndarray, list]:
        # Safety net: the generated-text path expects prompts already chat-templated
        # by encode_row. Warn if the template's (model-specific, derived) assistant
        # marker is absent — e.g. a custom encode_row produced raw prompts.
        marker = chat_template_assistant_marker(self._tokenizer)
        if marker and formatted_prompts_batch and marker not in formatted_prompts_batch[0]:
            logging.warning(
                f"Generated-text prompt does not contain the chat-template assistant marker "
                f"{marker!r}; encode_row may be producing un-templated (raw) prompts for this "
                f"chat model."
            )

        try:
            # The chat template (incl. system prompt) is applied upstream by
            # encode_row; here we just generate the raw completions.
            raw_generations = generate_text_batch(
                text_inputs=formatted_prompts_batch,
                model=self.model,
                tokenizer=self.tokenizer,
                context_size=context_size or self.context_size,
                max_new_tokens=self.max_new_tokens,
                temperature=self._resolve_temperature(question),
                seed=self.seed,
            )

            # `enable_thinking` drives only the post-processing (stripping the
            # `<think>` block); the template was applied upstream. Resolve the
            # `</think>` token id once for the token-id split. Mirrors vLLM's
            # `_risk_estimates_from_text` so both backends share one path.
            enable_thinking = reasoning_to_enable_thinking(self.reasoning)
            thinking_end_token_id = get_thinking_end_token_id(self._tokenizer)

            # Track regex extraction-failure rate for parity with the vLLM
            # backend (get_answer_from_model_output returns NaN on parse failure).
            risk_estimates_batch = []
            outputs: list = []
            for idx, (prompt, raw_generation) in enumerate(zip(formatted_prompts_batch, raw_generations)):
                generated_text = raw_generation["text"]
                output = _postprocess_generated_text(
                    enable_thinking=enable_thinking,
                    i=idx,
                    n=len(raw_generations),
                    text=generated_text,
                    token_ids=raw_generation["token_ids"],
                    tokenizer=self._tokenizer,
                    thinking_end_token_id=thinking_end_token_id,
                )
                response_text = output["response"]
                risk_estimate = question.get_answer_from_model_output(text=response_text)
                self._regex_total += 1
                if np.isnan(risk_estimate):
                    self._regex_failed += 1
                risk_estimates_batch.append(risk_estimate)
                outputs.append(output)
                self._maybe_warn_regex_extraction_failure_rate()
                self._maybe_log_generation(prompt, generated_text, risk_estimate)

            if self.token_tracker is not None:
                completion_tokens = sum(len(self._tokenizer.encode(o["response"], add_special_tokens=False)) for o in outputs)
                self.token_tracker.record_batch(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    batch_size=len(formatted_prompts_batch),
                )

            return np.asarray(risk_estimates_batch, dtype=float), outputs
        except Exception as error:
            logging.error(f"Error occurred while querying model: {error}")
            raise
