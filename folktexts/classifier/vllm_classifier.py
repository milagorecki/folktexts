"""Module for using vLLM as the local LLM-inference backend.

Mirrors `TransformersLLMClassifier` for the score-extraction contract — the
same QA decoders are reused, only the model-call inner loop changes.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from folktexts.llm_utils import (
    _postprocess_generated_text,
    decode_topk_logprobs_to_risk_estimate,
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

from .._utils import hash_dict
from .base import EncodeRowCallable, LLMClassifier

if TYPE_CHECKING:
    import vllm
    from vllm import SamplingParams

# Top-K logprobs to request from vLLM. WebAPI uses 20; we bump to 50 here
# because zero-shot prompts on instruction-tuned base models can push A/B
# answer letters into the long tail behind prose continuations
# ("I", "The", "Based"…). 50 covers those without materially changing
# results on models where A/B are clearly top-2.
_TOPK_LOGPROBS = 50


class VLLMClassifier(LLMClassifier):
    """Use a vLLM `LLM` engine to produce risk scores."""

    def __init__(
        self,
        llm: "vllm.LLM",
        tokenizer: AutoTokenizer,
        task: TaskMetadata | str,
        *,
        model_name_or_path: str | Path | None = None,
        encode_row: EncodeRowCallable = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        seed: int = 42,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier backed by vLLM.

        Parameters
        ----------
        llm : vllm.LLM
            A loaded vLLM engine. See `folktexts.llm_utils.load_vllm_model`.
        tokenizer : AutoTokenizer
            The HuggingFace tokenizer for the model. Usually obtained via
            `llm.get_tokenizer()`; passed in explicitly so observability hooks
            (chat-template helpers, vocab lookups) work without reaching into
            vLLM internals.
        task : TaskMetadata | str
            The task metadata object or name of an already created task.
        model_name_or_path : str | Path, optional
            The model path / name used to load the engine. Used for the
            display name and as a stable hash input. Defaults to the engine's
            internal `model_config.model` if available.
        encode_row, threshold, correct_order_bias, seed,
        **inference_kwargs
            Forwarded to `LLMClassifier`. See base-class docs.
        """
        self._llm = llm
        self._tokenizer = tokenizer

        # Resolve a stable name + vocab_dim without poking vLLM internals where
        # possible. AutoConfig is the canonical source of truth for vocab_size
        # (the logits axis) — same value `model.config.vocab_size` would give
        # on the transformers path.
        resolved_path = self._resolve_model_path(model_name_or_path)
        model_name = Path(resolved_path).name if resolved_path else "vllm-model"
        self._model_name_or_path = resolved_path
        self._vocab_dim = self._resolve_vocab_dim(resolved_path, tokenizer)

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

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(self, model_name_or_path: str | Path | None) -> str | None:
        """Best-effort lookup of the model path used to load the engine."""
        if model_name_or_path is not None:
            return str(model_name_or_path)
        # vLLM has shuffled this attribute across versions; try a few paths
        # before giving up.
        for getter in (
            lambda llm: llm.llm_engine.model_config.model,
            lambda llm: llm.llm_engine.get_model_config().model,
            lambda llm: llm.llm_engine.vllm_config.model_config.model,
        ):
            try:
                return str(getter(self._llm))
            except AttributeError:
                continue
        return None

    def _resolve_vocab_dim(
        self,
        model_name_or_path: str | None,
        tokenizer: AutoTokenizer,
    ) -> int:
        """Return the model's logits-axis vocab size.

        Always prefer `AutoConfig.vocab_size` over `len(tokenizer.get_vocab())`
        (the latter diverges across families — Gemma-3 has it == vocab_size+1,
        Llama-3.2 has it == vocab_size+256). See CLAUDE.md "Gotchas".
        """
        if model_name_or_path is not None:
            try:
                config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                )
                # Multimodal Gemma-3 and similar wrap the language-model config
                # under `text_config`; check both top-level and nested.
                vs = getattr(config, "vocab_size", None)
                if vs is None:
                    vs = getattr(getattr(config, "text_config", None), "vocab_size", None)
                if vs is not None:
                    return int(vs)
                logging.warning(
                    f"AutoConfig {type(config).__name__} for {model_name_or_path} "
                    "exposes no vocab_size at top level or text_config; falling back."
                )
            except Exception as exc:
                logging.warning(
                    f"AutoConfig.from_pretrained failed for {model_name_or_path}: "
                    f"{exc!r}; falling back to tokenizer-derived vocab size."
                )
        # Fallback: best of the two tokenizer-derived numbers; warn loudly,
        # since this is the path that the vocab-mismatch bug used to trip on.
        fallback = max(
            getattr(tokenizer, "vocab_size", 0),
            len(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else 0,
        )
        logging.warning(
            f"Falling back to tokenizer-derived vocab_dim={fallback}; "
            f"this can mis-size the logits mask on Gemma-3 / Llama-3.2 "
            f"families. Pass `model_name_or_path` explicitly to use AutoConfig."
        )
        return int(fallback)

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

    # ------------------------------------------------------------------
    # Properties / hashing
    # ------------------------------------------------------------------

    @property
    def llm(self):
        return self._llm

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def __hash__(self) -> int:
        """Hash distinct from `TransformersLLMClassifier.__hash__`.

        Including the backend tag prevents result-file paths
        (`results.bench-{hash}.json`) from colliding when the same model is
        run under both backends — predictions can differ on the order of 1e-3
        due to attention-kernel differences, so re-using a transformers CSV
        for a vLLM run would silently mix them.
        """
        hash_params = dict(
            super_hash=super().__hash__(),
            backend="vllm",
            vocab_dim=self._vocab_dim,
        )
        return int(hash_dict(hash_params), 16)

    # ------------------------------------------------------------------
    # Inference dispatch
    # ------------------------------------------------------------------

    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> tuple[np.ndarray, list]:
        """Query vLLM with a batch of prompts and return risk estimates."""
        # Generated-text types decode from a sampled response; base types decode
        # from token probabilities. (Decoding is a QA-type property, not a flag.)
        if question.use_generated_text:
            return self._risk_estimates_from_text(prompts_batch, question, context_size)

        # token-probability path
        risk_estimates = self._risk_estimates_from_logprobs(prompts_batch, question, context_size)
        # The base compute loop unpacks (risk_estimates, per-row outputs). The
        # token-probability paths surface no per-row metadata, so pad with None.
        return risk_estimates, [None] * len(risk_estimates)

    # ------------------------------------------------------------------
    # Text generation path: text generation + regex extraction
    # ------------------------------------------------------------------

    def _risk_estimates_from_text(
        self,
        formatted_prompts_batch: list[str],
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int | None,
    ) -> tuple[np.ndarray, list]:
        """Generated-text path: sample a full response and parse the answer.

        Mirrors the transformers backend's `use_generated_text` path — the chat
        template is already applied upstream by the encode_row function, so the
        prompts are generated as-is, then decoded via `get_answer_from_model_output`
        (numeric: probability regex; MCQ: answer-key regex). Handles both text
        QA types, and returns per-row output dicts so the base loop can record
        MCQ generations.
        """
        from vllm import (
            SamplingParams,  # local import — keeps module importable without vllm
        )

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

        # `enable_thinking` is derived from the reasoning kwarg only to post-process
        # the output (strip the `<think>` block); the template is applied upstream.
        enable_thinking = reasoning_to_enable_thinking(self.reasoning)

        sampling_params = SamplingParams(
            temperature=self._resolve_temperature(question),
            max_tokens=self.max_new_tokens,
            seed=self.seed,
        )
        raw_generations = self._llm.generate(formatted_prompts_batch, sampling_params)

        risk_estimates_batch: list[float] = []
        outputs: list = []
        for idx, (prompt, raw_generation) in enumerate(zip(formatted_prompts_batch, raw_generations)):
            completion = raw_generation.outputs[0]
            generated_text = completion.text
            # vLLM surfaces both the decoded text and the token ids; pass both so
            # the shared helper can honor the global split-mode toggle (token-id
            # mode degrades to string mode when token ids are absent, e.g. stubs).
            token_ids = getattr(completion, "token_ids", None)
            output = _postprocess_generated_text(
                enable_thinking=enable_thinking,
                i=idx,
                n=len(raw_generations),
                text=generated_text,
                token_ids=list(token_ids) if token_ids is not None else None,
                tokenizer=self._tokenizer,
            )
            response_text = output["response"]

            # `get_answer_from_model_output` returns NaN when the answer can't be
            # parsed; track that so a benchmark that mostly fails extraction
            # surfaces in the logs instead of silently dropping to NaN.
            risk_estimate = question.get_answer_from_model_output(text=response_text)
            self._regex_total += 1
            if np.isnan(risk_estimate):
                self._regex_failed += 1
            risk_estimates_batch.append(risk_estimate)
            outputs.append(output)
            self._maybe_warn_regex_extraction_failure_rate()
            self._maybe_log_generation(prompt, generated_text, risk_estimate)

        return np.asarray(risk_estimates_batch, dtype=float), outputs

    # ------------------------------------------------------------------
    # Token probability path: greedy next-token decoding
    # ------------------------------------------------------------------

    def _risk_estimates_from_logprobs(
        self,
        prompts_batch: list[str],
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int | None,
    ) -> np.ndarray:
        if isinstance(question, DirectNumericQA):
            sampling_params = self._sampling_params_numeric(question)
        else:
            sampling_params = self._sampling_params_multiple_choice(question)
        outputs = self._llm.generate(prompts_batch, sampling_params)
        risk_estimates_batch: list[float] = []
        for prompt, request_output in zip(prompts_batch, outputs):
            per_pass_topk = self._extract_per_pass_topk(request_output)
            risk_estimate = decode_topk_logprobs_to_risk_estimate(
                per_pass_topk,
                tokenizer_vocab=self._tokenizer.get_vocab(),
                vocab_dim=self._vocab_dim,
                question=question,
            )
            risk_estimates_batch.append(risk_estimate)
            if self._should_log_generation():
                # Decode the per-pass top-K token ids to strings + probabilities
                # for the logprob-path debug log (only when logging is enabled).
                # Invert the same `get_vocab()` the decode above uses (id -> token).
                id_to_token = {tid: tok for tok, tid in self._tokenizer.get_vocab().items()}
                self._maybe_log_logprobs(
                    prompt,
                    [
                        {id_to_token.get(tid, str(tid)): float(np.exp(lp)) for tid, lp in pass_topk.items()}
                        for pass_topk in per_pass_topk
                    ],
                    risk_estimate,
                )

        return np.asarray(risk_estimates_batch, dtype=float)

    # ------------------------------------------------------------------
    # DirectNumericQA path: greedy + digit-only constraint
    # ------------------------------------------------------------------

    def _sampling_params_numeric(self, question: DirectNumericQA) -> SamplingParams:
        from vllm import SamplingParams

        digit_token_ids = sorted(
            {
                tok_id
                for token, tok_id in self._tokenizer.get_vocab().items()
                if token.isdecimal() and 0 <= tok_id < self._vocab_dim
            }
        )
        if not digit_token_ids:
            raise RuntimeError("No digit tokens found in tokenizer vocabulary; cannot run DirectNumericQA on this model.")

        # Always 0.0 — DirectNumericQA reads the next-token distribution, it doesn't
        # sample. With `logprobs_mode="processed_logprobs"` any temperature > 0
        # would rescale the returned logprobs (vLLM divides logits by the
        # temperature before computing them) and silently change risk scores
        # relative to the other backends, which read untempered probabilities.
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=question.num_forward_passes,
            logprobs=_TOPK_LOGPROBS,
            allowed_token_ids=digit_token_ids,
            seed=self.seed,
        )

        return sampling_params

    # ------------------------------------------------------------------
    # MultipleChoiceQA path: unconstrained next-token logprobs
    # ------------------------------------------------------------------

    def _sampling_params_multiple_choice(self, question: MultipleChoiceQA) -> SamplingParams:
        # Match the transformers contract: MC reads the unconstrained next-token
        # softmax (no `allowed_token_ids` mask). The QA decoder's prefix-variant
        # logic + answer-token renormalisation handles candidates not in top-K
        # the same way it handles low-mass tokens on the transformers path.
        from vllm import SamplingParams

        # Always 0.0 — see the equivalent comment in `_risk_estimates_numeric`.
        # Output length reads `question.num_forward_passes` (= 1 for MC) uniformly
        # with the numeric path and transformers, rather than hardcoding it.
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=question.num_forward_passes,
            logprobs=_TOPK_LOGPROBS,
            seed=self.seed,
        )
        return sampling_params

    # ------------------------------------------------------------------
    # vLLM output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_per_pass_topk(request_output) -> list[dict[int, float]]:
        """Convert vLLM's per-position logprobs into ``[{token_id: logprob}, ...]``.

        ``request_output.outputs[0].logprobs`` is a list (one entry per
        generated token) of dicts ``{token_id: Logprob(logprob, ...)}``. We
        only need the ``logprob`` field; the rest is metadata.
        """
        completion = request_output.outputs[0]
        position_logprobs = completion.logprobs or []
        per_pass: list[dict[int, float]] = []
        for pos in position_logprobs:
            per_pass.append({int(tok_id): float(getattr(lp, "logprob", lp)) for tok_id, lp in pos.items()})
        return per_pass
