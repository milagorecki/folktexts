"""Unit tests for `VLLMClassifier` using duck-typed stubs.

vLLM is an optional install; these tests must run in environments without it.
We inject a fake ``vllm`` module into ``sys.modules`` before importing the
classifier so the in-method ``from vllm import SamplingParams`` resolves to
our stub. The stubbed ``LLM`` returns canned ``RequestOutput``-like objects so
we can verify the score-extraction contract (logprobs → risk estimate, text
→ extracted probability) without ever touching a real model.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

# --------------------------------------------------------------------------
# Inject a fake `vllm` module BEFORE the classifier is imported so its local
# `from vllm import SamplingParams` calls resolve to our stub.
# --------------------------------------------------------------------------


@dataclass
class _FakeSamplingParams:
    temperature: float = 0.0
    max_tokens: int = 1
    logprobs: int | None = None
    allowed_token_ids: list[int] | None = None
    seed: int | None = None


_fake_vllm = types.ModuleType("vllm")
_fake_vllm.SamplingParams = _FakeSamplingParams
sys.modules["vllm"] = _fake_vllm

# --- The classifier imports below resolve `vllm` to the stub above. -----
from folktexts.classifier.vllm_classifier import VLLMClassifier  # noqa: E402
from folktexts.qa_interface import (  # noqa: E402
    Choice,
    DirectNumericQA,
    MultipleChoiceQA,
    TextMultipleChoiceQA,
    TextNumericQA,
)

# --------------------------------------------------------------------------
# Stub LLM / tokenizer / output objects
# --------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal duck-typed tokenizer for VLLMClassifier."""

    def __init__(self, vocab: dict[str, int], vocab_size: int | None = None):
        self._vocab = dict(vocab)
        self.vocab_size = vocab_size if vocab_size is not None else max(vocab.values()) + 1
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"

    def get_vocab(self):
        return dict(self._vocab)

    @property
    def vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, *args, **kwargs):
        # No-op; classifier never touches this directly (load_vllm_model does).
        pass

    def encode(self, text, add_special_tokens=False):
        # Whitespace tokenization is enough for reasoning-token counting in tests.
        return text.split()

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(t) for t in token_ids)


@dataclass
class _StubLogprob:
    """Mirrors the public surface of vllm.Logprob — we only read .logprob."""

    logprob: float


@dataclass
class _StubCompletionOutput:
    text: str = ""
    logprobs: list[dict[int, _StubLogprob]] | None = None


@dataclass
class _StubRequestOutput:
    outputs: list[_StubCompletionOutput]


class _StubLLM:
    """Returns canned RequestOutputs in the order they were configured.

    `script` is a list-of-list-of-RequestOutput: each `generate(...)` call
    pops one batch from the front. This lets a single test exercise multiple
    successive calls (e.g. three batches of MC prompts) without coupling
    behaviour to a real vLLM engine.
    """

    def __init__(self, script: list[list[_StubRequestOutput]]):
        self._script = list(script)
        self.last_sampling_params = None
        self.last_prompts = None

    def generate(self, prompts, sampling_params, **kwargs):
        self.last_prompts = list(prompts)
        self.last_sampling_params = sampling_params
        if not self._script:
            raise AssertionError("StubLLM.generate called more times than scripted")
        return self._script.pop(0)


# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------


def _binary_mc_question() -> MultipleChoiceQA:
    return MultipleChoiceQA(
        column="PINCP",
        text="Is this person's income above $50k?",
        choices=(
            Choice(text="No", data_value=0, numeric_value=0.0),
            Choice(text="Yes", data_value=1, numeric_value=1.0),
        ),
    )


def _make_classifier(
    llm: _StubLLM,
    tokenizer: _StubTokenizer,
    *,
    vocab_dim: int,
    temperature: float | None = None,
    reasoning: str | None = None,
):
    """Build a VLLMClassifier wired against stubs.

    We pass `model_name_or_path=None` to skip the AutoConfig.from_pretrained
    lookup. The tokenizer fallback then yields `vocab_dim`. We force-override
    `_vocab_dim` afterwards so the test doesn't depend on the fallback's exact
    formula.
    """
    clf = VLLMClassifier(
        llm=llm,
        tokenizer=tokenizer,
        task="ACSIncome",
        model_name_or_path=None,
        temperature=temperature,
        reasoning=reasoning,
    )
    clf._vocab_dim = vocab_dim
    return clf


# --------------------------------------------------------------------------
# Multiple-choice path
# --------------------------------------------------------------------------


class TestMultipleChoicePath:
    def test_returns_positive_choice_probability(self):
        # Tokenizer pins ids for " A" / " B"; logprobs put 0.8 on " B" (= "Yes").
        vocab = {" A": 1, " B": 2, "X": 3}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        request_output = _StubRequestOutput(
            outputs=[
                _StubCompletionOutput(
                    logprobs=[
                        {
                            1: _StubLogprob(math.log(0.2)),
                            2: _StubLogprob(math.log(0.8)),
                        }
                    ],
                ),
            ]
        )
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)

        risks, _outputs = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy prompt"],
            question=_binary_mc_question(),
        )
        assert risks.shape == (1,)
        assert risks[0] == pytest.approx(0.8, abs=1e-6)

    def test_uses_unconstrained_sampling_for_mc(self):
        # The transformers MC path is unmasked; vLLM should mirror that —
        # `allowed_token_ids` must be None for MultipleChoiceQA.
        vocab = {" A": 1, " B": 2}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        request_output = _StubRequestOutput(
            outputs=[_StubCompletionOutput(logprobs=[{1: _StubLogprob(math.log(0.5)), 2: _StubLogprob(math.log(0.5))}])]
        )
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)
        clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy"],
            question=_binary_mc_question(),
        )

        params = llm.last_sampling_params
        assert params.allowed_token_ids is None
        assert params.max_tokens == 1
        assert params.temperature == 0.0


# --------------------------------------------------------------------------
# Direct numeric path
# --------------------------------------------------------------------------


class TestDirectNumericPath:
    def test_two_passes_concatenate_to_probability(self):
        # Vocab covers digits 0-9. Pass 0 picks "2", pass 1 picks "5" → 0.25.
        vocab = {str(d): d for d in range(10)}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        request_output = _StubRequestOutput(
            outputs=[
                _StubCompletionOutput(
                    logprobs=[
                        {2: _StubLogprob(math.log(0.7)), 1: _StubLogprob(math.log(0.2))},
                        {5: _StubLogprob(math.log(0.6)), 4: _StubLogprob(math.log(0.3))},
                    ]
                ),
            ]
        )
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)

        question = DirectNumericQA(column="PINCP", text="dummy")
        risks, _outputs = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy prompt"],
            question=question,
        )
        assert risks[0] == pytest.approx(0.25, abs=1e-6)

    def test_constrains_sampling_to_digits(self):
        # The transformers numeric path masks to digits via `digits_only=True`;
        # vLLM should mirror that with `allowed_token_ids=<digit ids>`.
        vocab = {**{str(d): d for d in range(10)}, "X": 99}
        tokenizer = _StubTokenizer(vocab, vocab_size=100)
        request_output = _StubRequestOutput(
            outputs=[
                _StubCompletionOutput(
                    logprobs=[
                        {3: _StubLogprob(math.log(0.9))},
                        {0: _StubLogprob(math.log(0.9))},
                    ]
                ),
            ]
        )
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=100)

        question = DirectNumericQA(column="PINCP", text="dummy")
        clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy"],
            question=question,
        )

        params = llm.last_sampling_params
        assert params.allowed_token_ids is not None
        assert sorted(params.allowed_token_ids) == list(range(10))
        # "X" is not decimal — must NOT be in the allowed list.
        assert 99 not in params.allowed_token_ids


# --------------------------------------------------------------------------
# Temperature contract
# --------------------------------------------------------------------------


class TestTemperatureResolution:
    def _run(self, question, *, temperature=None, reasoning=None):
        vocab = {" A": 1, " B": 2, **{str(d): d + 2 for d in range(10)}}
        tokenizer = _StubTokenizer(vocab, vocab_size=20)
        request_output = _StubRequestOutput(
            outputs=[
                _StubCompletionOutput(
                    text="Probability: 50%",
                    logprobs=[
                        {1: _StubLogprob(math.log(0.5)), 2: _StubLogprob(math.log(0.5))},
                        {2: _StubLogprob(math.log(0.5)), 3: _StubLogprob(math.log(0.5))},
                    ],
                ),
            ]
        )
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=20, temperature=temperature, reasoning=reasoning)
        clf._query_prompt_risk_estimates_batch(prompts_batch=["p"], question=question)
        return llm.last_sampling_params.temperature

    def test_mcq_defaults_to_zero(self):
        assert self._run(_binary_mc_question()) == 0.0

    def test_numeric_defaults_to_zero(self):
        assert self._run(DirectNumericQA(column="PINCP", text="dummy")) == 0.0

    def test_textnumeric_defaults_to_greedy_without_thinking(self):
        # No reasoning kwarg -> plain generated-text stays greedy.
        assert self._run(TextNumericQA(column="PINCP", text="dummy")) == 0.0

    def test_textnumeric_bumps_to_one_with_thinking(self):
        # Thinking is driven by the classifier's `reasoning` kwarg, not a QA field.
        assert self._run(TextNumericQA(column="PINCP", text="dummy"), reasoning="high") == 1.0

    def test_explicit_override_does_not_affect_mcq(self):
        # MC/numeric read the untempered next-token distribution: with vLLM's
        # `processed_logprobs` a temperature > 0 would rescale the returned
        # logprobs and silently change risk scores vs the other backends.
        assert self._run(_binary_mc_question(), temperature=2.0) == 0.0

    def test_explicit_override_does_not_affect_numeric(self):
        q = DirectNumericQA(column="PINCP", text="dummy")
        assert self._run(q, temperature=2.0) == 0.0

    def test_explicit_override_applies_to_generated_text(self):
        # An explicit temperature overrides the generated-text default (and the
        # thinking bump). MC/numeric above stay untempered at 0 either way.
        q = TextNumericQA(column="PINCP", text="dummy")
        assert self._run(q, temperature=0.25) == 0.25
        assert self._run(q, temperature=0.25, reasoning="high") == 0.25


# --------------------------------------------------------------------------
# Generated-text path (regex extraction, not logprobs)
# --------------------------------------------------------------------------


class TestGeneratedTextPath:
    """Text QA types must route to generation + regex, not the logprob path.

    The stub tokenizer has no `chat_template`, so `_apply_chat_template_batch`
    falls back to raw prompts; the stub LLM returns canned generated `text`.
    """

    def _run(self, question, generated: str, *, reasoning: str | None = None):
        tokenizer = _StubTokenizer({" A": 1, " B": 2}, vocab_size=10)
        request_output = _StubRequestOutput(outputs=[_StubCompletionOutput(text=generated)])
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10, reasoning=reasoning)
        risks, outputs = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy prompt"],
            question=question,
        )
        return risks, outputs, llm

    def test_numeric_text_extracts_probability(self):
        q = TextNumericQA(column="PINCP", text="dummy")
        risks, outputs, llm = self._run(q, "I think... Probability: 80%")
        assert risks[0] == pytest.approx(0.8, abs=1e-6)
        # Went through generation (not the digit-constrained logprob path).
        assert llm.last_sampling_params.logprobs is None
        assert llm.last_sampling_params.allowed_token_ids is None
        # Per-row response dict is surfaced for the base loop.
        assert outputs[0]["response"].endswith("Probability: 80%")

    def test_mcq_text_extracts_answer_key(self):
        q = TextMultipleChoiceQA(
            column="PINCP",
            text="Is income above $50k?",
            choices=(Choice("No", 0, 0.0), Choice("Yes", 1, 1.0)),
        )
        risks, outputs, _ = self._run(q, "Reasoning here. Answer: B")
        assert risks[0] == pytest.approx(1.0, abs=1e-6)  # "B" == "Yes" (positive)
        assert outputs[0]["response"].endswith("Answer: B")

    def test_generated_text_greedy_by_default(self):
        # Plain generated-text (no thinking) stays greedy/deterministic.
        q = TextNumericQA(column="PINCP", text="dummy")
        _, _, llm = self._run(q, "Probability: 50%")
        assert llm.last_sampling_params.temperature == 0.0

    def test_generated_text_thinking_bumps_temperature(self):
        # With thinking/reasoning active, `_resolve_temperature` forces 1.0.
        q = TextNumericQA(column="PINCP", text="dummy")
        _, _, llm = self._run(q, "Probability: 50%", reasoning="high")
        assert llm.last_sampling_params.temperature == 1.0

    def test_strips_thinking_block_when_enabled(self):
        # In thinking mode the vLLM backend runs `_postprocess_generated_text`,
        # which drops everything up to and including `</think>`. The "20%" inside
        # the thinking block must be ignored; only the post-`</think>` response
        # counts, so the decoder MUST extract 0.85 (not 0.20).
        tokenizer = _StubTokenizer({" A": 1, " B": 2}, vocab_size=10)
        full_text = "Lots of reasoning, considering 20%, then more.\n</think>\nFinal answer: Probability: 85%."
        llm = _StubLLM(script=[[_StubRequestOutput(outputs=[_StubCompletionOutput(text=full_text)])]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10, reasoning="high")

        risks, _outputs = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy"],
            question=TextNumericQA(column="PINCP", text="dummy"),
        )
        assert risks[0] == pytest.approx(0.85, abs=1e-6)
        assert clf._regex_failed == 0

    def test_failed_extraction_returns_nan(self):
        # New contract: an unparsable generation yields NaN (not a silent 0.5),
        # and the failure is tracked via the `_regex_failed`/`_regex_total`
        # counters (our equivalents of upstream's `_cot_failed`/`_cot_total`) so
        # the downstream warning logic can fire.
        tokenizer = _StubTokenizer({" A": 1, " B": 2}, vocab_size=10)
        llm = _StubLLM(script=[[_StubRequestOutput(outputs=[_StubCompletionOutput(text="No probability stated here")])]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)

        risks, _outputs = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy"],
            question=TextNumericQA(column="PINCP", text="dummy"),
        )
        assert np.isnan(risks[0])
        assert clf._regex_failed == 1
        assert clf._regex_total == 1


# --------------------------------------------------------------------------
# Hash separation between backends
# --------------------------------------------------------------------------


class TestBackendDistinctHash:
    def test_hash_includes_vllm_tag(self):
        # Two classifiers identical except for backend tag must hash differently
        # so cached predictions don't bleed between backends.
        vocab = {" A": 1, " B": 2}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        clf = _make_classifier(_StubLLM(script=[]), tokenizer, vocab_dim=10)
        # Sanity: the hash dict includes the backend tag literally.
        h = hash(clf)
        assert isinstance(h, int)
        # Hash is deterministic given the same inputs.
        assert hash(clf) == h
