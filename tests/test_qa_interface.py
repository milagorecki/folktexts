"""Unit tests for `qa_interface`.

Structural properties this file guards:

1. `get_question_prompt(with_answer_prefill: bool)` — `True` (default) keeps
   the legacy zero-shot string byte-for-byte; `False` removes the answer
   prefill so the chat-template path can supply it as the assistant turn
   without duplicating it in the user message.

2. `get_answer_prefix()` — the answer prefill string returned by each QA
   subclass independently of the full question prompt.

3. `_get_numeric_tokens(tokenizer_vocab, vocab_dim)` — filters digit / decimal
   tokens whose ids fall outside `[0, vocab_dim)`. The caller (`get_answer_from_model_output`)
   derives `vocab_dim` from the actual logits axis (`last_token_probs.shape[-1]`),
   so out-of-range vocab entries no longer trip an `IndexError` deep in the
   probability lookup.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from folktexts.classifier.base import InferenceConfig, LLMClassifier
from folktexts.qa_interface import (
    Choice,
    DirectNumericQA,
    MultipleChoiceQA,
    TextMultipleChoiceQA,
    TextNumericQA,
)

# ----------------------------------------------------------------------
# DirectNumericQA.get_question_prompt
# ----------------------------------------------------------------------


class TestDirectNumericQAGetQuestionPrompt:
    def _q(self, answer_probability: bool = True) -> DirectNumericQA:
        return DirectNumericQA(
            column="PINCP",
            text="What is this person's estimated yearly income?",
            answer_probability=answer_probability,
        )

    def test_default_matches_legacy_zero_shot_string(self):
        # Pin byte-for-byte equality with the pre-refactor zero-shot output
        # so the paper-reproducing path is provably untouched.
        expected = "Question: What is this person's estimated yearly income?\nAnswer (between 0 and 1): 0."
        assert self._q().get_question_prompt() == expected
        assert self._q().get_question_prompt(with_answer_prefill=True) == expected

    def test_with_answer_prefill_false_omits_prefill(self):
        out = self._q().get_question_prompt(with_answer_prefill=False)
        assert "Answer (between 0 and 1)" not in out
        assert "0." not in out
        # The bare question must still be present and the string must end at
        # the question text — that's exactly what the chat user-turn needs.
        assert out == "Question: What is this person's estimated yearly income?"

    def test_answer_probability_false_with_prefill_emits_open_answer(self):
        # `answer_probability=False` ⇒ open-ended numeric Q&A; the prefill is
        # `"Answer: "` (trailing space matters for tokenization in the
        # zero-shot path).
        out = self._q(answer_probability=False).get_question_prompt()
        assert out.endswith("\nAnswer: ")

    def test_answer_probability_false_no_prefill_omits_answer_line(self):
        out = self._q(answer_probability=False).get_question_prompt(
            with_answer_prefill=False,
        )
        assert "Answer" not in out
        assert out == "Question: What is this person's estimated yearly income?"


# ----------------------------------------------------------------------
# MultipleChoiceQA.get_question_prompt
# ----------------------------------------------------------------------


class TestMultipleChoiceQAGetQuestionPrompt:
    def _q(self) -> MultipleChoiceQA:
        return MultipleChoiceQA(
            column="PINCP",
            text="Is this person's income above $50k?",
            choices=(
                Choice(text="No", data_value=0, numeric_value=0.0),
                Choice(text="Yes", data_value=1, numeric_value=1.0),
            ),
        )

    def test_default_matches_legacy_zero_shot_string(self):
        expected = "Question: Is this person's income above $50k?\nA. No.\nB. Yes.\nAnswer:"
        assert self._q().get_question_prompt() == expected
        assert self._q().get_question_prompt(with_answer_prefill=True) == expected

    def test_with_answer_prefill_false_omits_answer_line(self):
        out = self._q().get_question_prompt(with_answer_prefill=False)
        # Choices must remain — they're the question content, not the prefill.
        assert "A. No." in out
        assert "B. Yes." in out
        # The trailing "Answer:" prefill is the only thing that should drop.
        assert not out.rstrip().endswith("Answer:")
        assert out == ("Question: Is this person's income above $50k?\nA. No.\nB. Yes.")


# ----------------------------------------------------------------------
# get_answer_prefix — the answer prefill string for each QA subclass
# ----------------------------------------------------------------------


class TestGetAnswerPrefix:
    def test_numeric_answer_probability_true(self):
        q = DirectNumericQA(column="x", text="dummy", answer_probability=True)
        assert q.get_answer_prefix() == "Answer (between 0 and 1): 0."

    def test_numeric_answer_probability_false(self):
        q = DirectNumericQA(column="x", text="dummy", answer_probability=False)
        assert q.get_answer_prefix() == "Answer: "

    def test_mc_answer_prefix(self):
        q = MultipleChoiceQA(
            column="x",
            text="dummy",
            choices=(
                Choice(text="No", data_value=0, numeric_value=0.0),
                Choice(text="Yes", data_value=1, numeric_value=1.0),
            ),
        )
        assert q.get_answer_prefix() == "Answer:"


# ----------------------------------------------------------------------
# DirectNumericQA._get_numeric_tokens — vocab_dim filter
# ----------------------------------------------------------------------


class TestGetNumericTokensVocabDimFilter:
    """The filter prevents an `IndexError` on tokenizers (e.g. Gemma-3) where
    digit-or-decimal-named tokens can sit at ids beyond `model.config.vocab_size`
    — the actual logits axis the caller indexes into.
    """

    def _q(self) -> DirectNumericQA:
        return DirectNumericQA(column="x", text="dummy")

    def test_drops_digit_token_whose_id_is_beyond_vocab_dim(self):
        vocab = {str(i): i for i in range(10)}
        vocab["888"] = 100  # out-of-range multi-digit
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=60)
        assert "888" not in nums
        # All single-digit base tokens (ids 0-9) are in-range and kept.
        assert all(str(i) in nums for i in range(10))

    def test_keeps_in_range_multi_digit_tokens(self):
        vocab = {str(i): i for i in range(10)}
        vocab["999"] = 50  # in-range
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=60)
        assert nums.get("999") == 50

    def test_keeps_decimal_when_in_range(self):
        vocab = {str(i): i for i in range(10)}
        vocab["."] = 7
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=60)
        assert nums.get(".") == 7

    def test_drops_decimal_when_out_of_range(self):
        vocab = {str(i): i for i in range(10)}
        vocab["."] = 100  # beyond vocab_dim
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=60)
        assert "." not in nums

    def test_no_decimal_in_vocab_is_handled(self):
        vocab = {str(i): i for i in range(10)}
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=10)
        assert "." not in nums
        # All digit tokens kept.
        assert len(nums) == 10

    def test_vocab_dim_zero_filters_everything(self):
        # Edge case: pathological vocab_dim. Should not crash, just drop all.
        vocab = {str(i): i for i in range(10)}
        vocab["."] = 7
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=0)
        assert nums == {}

    def test_returned_ids_are_all_within_vocab_dim(self):
        """Strong invariant: every returned id is a legal index into a
        `last_token_probs` array of length `vocab_dim`."""
        vocab = {str(i): i for i in range(20)}
        vocab["100"] = 30
        vocab["200"] = 70  # out-of-range
        vocab["."] = 50  # out-of-range
        vocab_dim = 40
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=vocab_dim)
        assert all(0 <= tid < vocab_dim for tid in nums.values())


# ----------------------------------------------------------------------
# DirectNumericQA.get_answer_from_model_output — derives vocab_dim from probs
# ----------------------------------------------------------------------


class TestGetAnswerFromModelOutputDerivesVocabDim:
    """The caller no longer has to thread `vocab_dim` in: it is derived from
    `last_token_probs.shape[-1]` (the logits axis the caller already gave us).
    This test pins that contract and that the filter is applied — i.e. an
    out-of-range digit-named token doesn't IndexError when scoring.
    """

    def _q(self) -> DirectNumericQA:
        return DirectNumericQA(column="x", text="dummy")

    def test_derives_vocab_dim_from_probs_and_drops_oor_tokens(self):
        # Vocab declares "5" at id 50 (in-range) and "9" at id 100 (out-of-range
        # for a probs array of width 60). Must not IndexError; "9" must not be
        # considered when picking the most-likely numeric token.
        vocab = {str(i): i for i in range(10)}  # "0"..."9" → ids 0..9 (in-range)
        vocab["99"] = 100  # out-of-range — the filter must drop this
        vocab["."] = 7
        # Probs array of width 60: assign all mass to id 5 on pass 0 ("5") and
        # id 3 on pass 1 ("3"). Expected answer: 0.53.
        probs = np.zeros((2, 60))
        probs[0, 5] = 1.0
        probs[1, 3] = 1.0
        ans = self._q().get_answer_from_model_output(probs, vocab)
        assert ans == pytest.approx(0.53)

    def test_no_indexerror_when_vocab_extends_past_logits_axis(self):
        # Gemma-style: digit-named tokens may exist at ids >= logits_dim. The
        # filter must drop them so `ltp[token_id]` doesn't IndexError.
        vocab = {str(i): i for i in range(10)}  # ids 0..9 — in-range
        vocab["77"] = 10  # digit-named, AT the logits axis — must be dropped
        probs = np.zeros((1, 10))
        probs[0, 4] = 1.0
        # `answer_probability=True` ⇒ "0." prefill + "4" → 0.4.
        ans = self._q().get_answer_from_model_output(probs, vocab)
        assert ans == pytest.approx(0.4)


# ----------------------------------------------------------------------
# QA-type consistency matrix
#
# The decode axis (token-probability vs generated-text) is expressed by the QA
# *type*, not a runtime flag. This matrix pins the invariants across all four
# types so any future drift (e.g. a scoring prefill leaking onto a text type)
# fails a test.
# ----------------------------------------------------------------------

_CHOICES = (Choice("No", 0, 0.0), Choice("Yes", 1, 1.0))


def _make_qa(cls):
    if issubclass(cls, MultipleChoiceQA):
        return cls(column="X", text="Q?", num_forward_passes=1, choices=_CHOICES)
    return cls(column="X", text="Q?")


# (type, is_text, is_numeric)
_QA_MATRIX = [
    (DirectNumericQA, False, True),
    (MultipleChoiceQA, False, False),
    (TextNumericQA, True, True),
    (TextMultipleChoiceQA, True, False),
]


@pytest.mark.parametrize("cls,is_text,is_numeric", _QA_MATRIX)
class TestQATypeConsistency:
    def test_use_generated_text_matches_type(self, cls, is_text, is_numeric):
        assert _make_qa(cls).use_generated_text is is_text

    def test_chat_prefill_present_iff_scoring(self, cls, is_text, is_numeric):
        # The scoring prefill (assistant turn) exists only on token-probability
        # types; generated-text types must carry none (it would fight the
        # answer-format instruction).
        assert (_make_qa(cls).default_chat_prompt is None) is is_text

    def test_format_instruction_present_iff_text(self, cls, is_text, is_numeric):
        # `format_instruction` lives on the generated-text mixin only; scoring
        # types don't declare it, so read defensively.
        assert (getattr(_make_qa(cls), "format_instruction", None) is not None) is is_text

    def test_system_prompt_appends_format_instruction_iff_text(self, cls, is_text, is_numeric):
        q = _make_qa(cls)
        sys_prompt = q.get_default_system_prompt()
        if is_text:
            assert q.format_instruction in sys_prompt
        else:
            assert sys_prompt == q.default_system_prompt

    def test_answer_prefix_empty_iff_text(self, cls, is_text, is_numeric):
        # Generated-text types have no scoring prefill, so no answer prefix.
        assert (_make_qa(cls).get_answer_prefix() == "") is is_text

    def test_answer_prefill_absent_iff_text(self, cls, is_text, is_numeric):
        # Generated-text types never bake an answer prefill into the question
        # (the model must produce free-form text to parse); scoring types do.
        base_cls = DirectNumericQA if is_numeric else MultipleChoiceQA
        base_prefix = _make_qa(base_cls).get_answer_prefix()
        prompt = _make_qa(cls).get_question_prompt().rstrip()
        assert prompt.endswith(base_prefix) is (not is_text)

    def test_model_output_decodes_via_the_right_path(self, cls, is_text, is_numeric):
        q = _make_qa(cls)
        if is_text:
            # Text types decode from `text`; passing only token probs must fail.
            answer = q.get_answer_from_model_output(text=("Probability: 80%" if is_numeric else "Answer: B"))
            assert answer == pytest.approx(0.8 if is_numeric else 1.0, abs=1e-6)
            with pytest.raises(ValueError):
                q.get_answer_from_model_output(text=None)
        else:
            # Scoring types decode from token probs; passing only `text` must fail.
            with pytest.raises(ValueError):
                q.get_answer_from_model_output(text="Probability: 80%")


# ======================================================================
# Generated-text QA types (TextNumericQA / TextMultipleChoiceQA)
#
# Port of upstream's `test_cot_qa.py` (`ChainOfThoughtQA`) to this branch's
# terminology: upstream's single CoT type maps to `TextNumericQA` (numeric
# answer + generated-text decoding); chain-of-thought is the `use_cot=True`
# modifier. Contract differences asserted here as the *new* behaviour:
#   * extraction failure -> NaN (not a silent 0.5)
#   * the extraction anchor lives in the *system prompt*, not the question
#   * plain generated-text is greedy; the thinking bump is in the classifier
# ======================================================================


@pytest.fixture
def text_numeric_qa() -> TextNumericQA:
    return TextNumericQA(
        column="PINCP",
        text="What is this person's estimated yearly income?",
    )


class TestTextNumericQA:
    # Numeric answer + generated-text decoding. Mirrors TestTextMultipleChoiceQA.

    # --- question prompt: `with_answer_prefill` kwarg accepted for interface
    # compatibility (Liskov substitution) but ignored — generated-text prompts
    # carry no answer prefill to strip ----------------------------------------
    def test_returns_non_empty_string(self, text_numeric_qa: TextNumericQA):
        assert isinstance(text_numeric_qa.get_question_prompt(), str)
        assert text_numeric_qa.get_question_prompt().strip()

    def test_with_answer_prefill_kwarg_accepted(self, text_numeric_qa: TextNumericQA):
        with_prefill = text_numeric_qa.get_question_prompt(with_answer_prefill=True)
        without_prefill = text_numeric_qa.get_question_prompt(with_answer_prefill=False)
        assert with_prefill == without_prefill

    def test_question_prompt_excludes_extraction_anchor(self, text_numeric_qa: TextNumericQA):
        # The "Probability: X%" anchor lives in the system prompt, not the question.
        assert "Probability: X%" not in text_numeric_qa.get_question_prompt()

    # --- system prompt: format instruction + optional CoT --------------------
    def test_system_prompt_includes_extraction_anchor(self, text_numeric_qa: TextNumericQA):
        # The system prompt must carry the "Probability: X%" anchor so the regex
        # extractor has a consistent target. If this anchor changes, the numeric
        # extraction patterns must be updated in lockstep.
        assert "Probability: X%" in text_numeric_qa.get_default_system_prompt()

    def test_cot_instruction_absent_by_default(self, text_numeric_qa: TextNumericQA):
        assert text_numeric_qa.use_cot is False
        assert text_numeric_qa.cot_instruction not in text_numeric_qa.get_default_system_prompt()

    def test_cot_present_and_precedes_format_when_enabled(self):
        # Reason first, then emit the parser-friendly answer: the CoT text must
        # appear before the format instruction in the assembled system prompt.
        q = TextNumericQA(column="PINCP", text="income?", use_cot=True)
        sys_prompt = q.get_default_system_prompt()
        assert q.cot_instruction in sys_prompt
        assert sys_prompt.index(q.cot_instruction) < sys_prompt.index(q.format_instruction)

    def test_use_cot_changes_identity(self):
        # `use_cot` is a real field, so it participates in QA identity (hash),
        # keeping cached results distinct.
        plain = TextNumericQA(column="PINCP", text="income?", use_cot=False)
        cot = TextNumericQA(column="PINCP", text="income?", use_cot=True)
        assert hash(plain) != hash(cot)

    # --- extract_probability_from_text: the regex pyramid (matches upstream) --
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Probability: 75%", 0.75),
            ("probability: 75%", 0.75),
            ("Probability: 0.75", 0.75),
            ("Probability is 80%", 0.80),
            ("Probability of 0.42", 0.42),
            ("After thinking: 25%", 0.25),
            ("The answer is 50 percent.", 0.50),
            ("My estimate is 0.33", 0.33),
        ],
    )
    def test_extracts_supported_formats(self, text: str, expected: float):
        assert TextNumericQA.extract_probability_from_text(text) == pytest.approx(expected, abs=1e-6)

    def test_uses_last_explicit_match(self):
        # The model may revise its estimate mid-reasoning; we trust the final
        # "Probability: X" line.
        text = "Probability: 30%. Wait, on reflection, Probability: 65%."
        assert TextNumericQA.extract_probability_from_text(text) == pytest.approx(0.65)

    def test_returns_none_when_no_signal(self):
        assert TextNumericQA.extract_probability_from_text("the answer is unclear") is None

    def test_rejects_out_of_range_probability(self):
        # A bare 150% must not be treated as a probability via the explicit
        # "Probability:" pattern; the function falls through and returns None.
        assert TextNumericQA.extract_probability_from_text("Probability: 150%") is None

    # --- get_answer_from_model_output: failure -> NaN (not upstream's 0.5) ----
    def test_returns_extracted_value(self, text_numeric_qa: TextNumericQA):
        assert text_numeric_qa.get_answer_from_model_output(text="Probability: 80%") == pytest.approx(0.80)

    def test_returns_nan_on_extraction_failure(self, text_numeric_qa: TextNumericQA):
        # NaN (not 0.5) marks a parse failure so the caller can distinguish it
        # from a genuine 0.5 and drop / hedge / impute it downstream.
        result = text_numeric_qa.get_answer_from_model_output(text="nonsense with no probability")
        assert np.isnan(result)

    def test_requires_text(self, text_numeric_qa: TextNumericQA):
        # Generated-text types decode from `text`, never from token probs.
        with pytest.raises(ValueError, match="text must be provided"):
            text_numeric_qa.get_answer_from_model_output()

    def test_default_temperature_is_greedy(self, text_numeric_qa: TextNumericQA):
        assert text_numeric_qa.default_temperature == 0.0


class TestGeneratedTextTemperature:
    # Plain generated-text is greedy (per-type default asserted in the type
    # classes); the bump to 1.0 under thinking is resolved by
    # `LLMClassifier._resolve_temperature`, not baked into the QA type.
    def test_token_probability_modes_stay_greedy(self):
        assert DirectNumericQA(column="PINCP", text="dummy").default_temperature == 0.0
        mcq = MultipleChoiceQA(
            column="PINCP",
            text="dummy",
            choices=(Choice("Yes", 1), Choice("No", 0)),
        )
        assert mcq.default_temperature == 0.0

    @pytest.mark.parametrize(
        "reasoning,expected",
        [
            (None, 0.0),  # plain, non-reasoning model -> greedy
            ("0", 1.0),  # reasoning-capable model, thinking toggled off -> still sampled
            ("high", 1.0),  # thinking on -> forced sampling
            ("low", 1.0),
            ("512", 1.0),  # Claude budget-tokens -> thinking on
        ],
    )
    def test_resolve_temperature_bumps_when_reasoning_set(self, text_numeric_qa, reasoning, expected):
        # `_resolve_temperature` only reads `self._inference`, so a duck-typed
        # stub with an InferenceConfig exercises the contract without a real model.
        stub = SimpleNamespace(_inference=InferenceConfig(temperature=None, reasoning=reasoning))
        assert LLMClassifier._resolve_temperature(stub, text_numeric_qa) == expected

    def test_explicit_override_wins_over_reasoning(self, text_numeric_qa):
        stub = SimpleNamespace(_inference=InferenceConfig(temperature=0.3, reasoning="high"))
        assert LLMClassifier._resolve_temperature(stub, text_numeric_qa) == 0.3


@pytest.fixture
def text_mcq() -> TextMultipleChoiceQA:
    # Binary choice: A -> "No" (0.0), B -> "Yes" (1.0). Risk = P(positive) so
    # picking "Yes" (B) yields 1.0 and "No" (A) yields 0.0.
    return TextMultipleChoiceQA(
        column="PINCP",
        text="Is this person's income above $50k?",
        choices=(Choice("No", 0, 0.0), Choice("Yes", 1, 1.0)),
    )


class TestTextMultipleChoiceQA:
    # The multiple-choice sibling of TextNumericQA: same generated-text contract,
    # but the answer is a choice letter (`Answer: X`) mapped to a risk estimate.
    def test_question_prompt_lsp_kwarg_identical(self, text_mcq):
        assert text_mcq.get_question_prompt(with_answer_prefill=True) == text_mcq.get_question_prompt(
            with_answer_prefill=False
        )

    def test_question_prompt_excludes_answer_anchor(self, text_mcq):
        # The "Answer: X" anchor lives in the system prompt, not the question.
        assert "Answer: X" not in text_mcq.get_question_prompt()

    def test_system_prompt_includes_answer_anchor(self, text_mcq):
        assert "Answer: X" in text_mcq.get_default_system_prompt()

    def test_cot_absent_by_default(self, text_mcq):
        assert text_mcq.use_cot is False
        assert text_mcq.cot_instruction not in text_mcq.get_default_system_prompt()

    def test_cot_present_and_precedes_format_when_enabled(self):
        q = TextMultipleChoiceQA(
            column="PINCP",
            text="q?",
            choices=(Choice("No", 0, 0.0), Choice("Yes", 1, 1.0)),
            use_cot=True,
        )
        sys_prompt = q.get_default_system_prompt()
        assert q.cot_instruction in sys_prompt
        assert sys_prompt.index(q.cot_instruction) < sys_prompt.index(q.format_instruction)

    @pytest.mark.parametrize(
        "text,expected_risk",
        [
            ("Answer: B", 1.0),  # B == "Yes" (positive)
            ("Answer: A", 0.0),  # A == "No"
            ("Lots of reasoning here.\nAnswer: B", 1.0),  # anchored, after a trace
        ],
    )
    def test_extracts_answer_key(self, text_mcq, text, expected_risk):
        assert text_mcq.get_answer_from_model_output(text=text) == pytest.approx(expected_risk)

    def test_uses_last_same_tier_match(self, text_mcq):
        # Both mentions are the same (key + trailing punctuation) tier, so the
        # last one wins — the model's revised final answer.
        text = "First I leaned Answer: A. On reflection, Answer: B."
        assert text_mcq.get_answer_from_model_output(text=text) == pytest.approx(1.0)

    def test_returns_nan_on_extraction_failure(self, text_mcq):
        # No answer key and no choice-text word present -> NaN (not a silent 0.5).
        assert np.isnan(text_mcq.get_answer_from_model_output(text="the outcome is unclear"))

    @pytest.mark.parametrize("text", ["I cannot determine this", "nothing is certain here"])
    def test_choice_word_inside_other_word_does_not_match(self, text_mcq, text):
        # Regression: the unanchored choice-text tier is word-bounded, so the
        # short choice "No" must not match inside "cannot"/"nothing" -> NaN.
        assert np.isnan(text_mcq.get_answer_from_model_output(text=text))

    def test_unanchored_standalone_choice_word_still_matches(self, text_mcq):
        # Word boundaries keep genuine standalone choice words working.
        assert text_mcq.get_answer_from_model_output(text="The answer is Yes") == pytest.approx(1.0)

    def test_requires_text(self, text_mcq):
        with pytest.raises(ValueError, match="text must be provided"):
            text_mcq.get_answer_from_model_output()

    def test_default_temperature_is_greedy(self, text_mcq):
        assert text_mcq.default_temperature == 0.0
