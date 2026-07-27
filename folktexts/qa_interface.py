"""Interface for question-answering with LLMs.

- Create different types of questions (direct numeric, multiple-choice).
- Encode questions and decode model outputs.
- Compute risk-estimate from model outputs.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import re
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Iterator

import numpy as np

from ._utils import hash_dict

# Minimum probability density assigned to all valid answers
# > small models will be worse at using valid answers...
ANSWER_PROB_THRESHOLD = 0.1

# Default answer keys for multiple-choice questions
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_ANSWER_PATTERNS = [
    # matches "Answer", followed by optional ":" and zero or more whitespaces
    r"[Aa]nswer:?\s*"
]

# ---------------------------------------------------------------------------
# Default system / chat prompts — owned here so each QAInterface subclass can
# declare its own defaults without importing from prompting.py (which imports
# from this module, which would create a circular dependency).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant. You answer multiple-choice questions \
based on the information provided. Respond with a single answer choice.
"""

NUMERIC_SYSTEM_PROMPT = """\
You are a helpful assistant. You provide numeric probability \
estimates based on the information provided.
"""
ANTHROPIC_CHAT_PROMPT = "If had to select one of the options, my answer would be"
GEMMA_CHAT_PROMPT = "The provided information suggests that the answer is"


# NOTE: The leading `0.` is part of the prefill, so the model only generates
# the digits after the decimal point. This caps the expressible probability
# at the open interval [0, 1) — true posteriors at or near 1.0 cannot be
# emitted exactly. If you need full [0, 1] coverage, override `chat_prompt`
# with e.g. `"Answer (between 0 and 1): "` and let the model produce the
# leading digit itself (note that this also widens the digit-scoring search
# space and may degrade calibration for low-probability cases).
NUMERIC_CHAT_PROMPT = "Answer (between 0 and 1): 0."

# Answer-format instructions appended to the system prompt on the generated-text
# path so the model emits output the text parsers can extract. Only applied when
# `use_generated_text=True`: the token-probability path reads logprobs directly
# and relies on the chat-prompt prefill instead, so a format instruction there
# would only fight that prefill. Each string matches its parser (MCQ ->
# `_ANSWER_PATTERNS` "Answer:"; numeric -> the "Probability: X%" parser in
# `TextNumericQA.extract_probability_from_text`).
MCQ_GENERATED_TEXT_FORMAT = (
    "Your response MUST end with your answer choice in the following format:\n"
    "Answer: X\n"
    "where X is the letter of your chosen option."
)
NUMERIC_GENERATED_TEXT_FORMAT = (
    "Your response MUST end with your probability estimate in the following format:\n"
    "Probability: X%\n"
    "where X is a number between 0 and 100."
)

COT_INSTRUCTION = (
    "Think step-by-step about the factors that could influence the answer to this question. "
    "After reasoning through the relevant information, provide your final answer."
)


@dataclass(frozen=True)
class QAInterface(ABC):
    """An interface for a question-answering system."""

    column: str
    text: str
    num_forward_passes: int
    use_generated_text: ClassVar[bool] = False

    # Subclasses override these to declare their mode-appropriate defaults.
    # `None` means "no default" (i.e. no system prompt / no chat prefill).
    default_system_prompt: ClassVar[str | None] = SYSTEM_PROMPT
    default_chat_prompt: ClassVar[str | None] = ANTHROPIC_CHAT_PROMPT

    # Default sampling temperature for *text-generation* prompting, read via
    # `LLMClassifier._resolve_temperature`. Greedy (0.0) by default — plain
    # generated-text stays deterministic/reproducible; `_resolve_temperature`
    # bumps this to 1.0 when thinking/reasoning is active (greedy is unreliable
    # or rejected by thinking models). Only meaningful for text generation:
    # token-probability methods (multiple-choice, direct-numeric) read the
    # untempered next-token distribution on every backend and never sample, so
    # temperature does not apply to them. A `ClassVar` (not a dataclass field)
    # so it does not affect the frozen dataclass hash / result-cache identity.
    default_temperature: ClassVar[float] = 0.0

    def get_default_system_prompt(self) -> str | None:
        """Default system prompt for this question.

        The token-probability path uses the plain system prompt (it reads
        logprobs and relies on the chat-prompt prefill instead). The
        generated-text types (`GeneratedTextQA`) override this to append the
        answer-format (and optional chain-of-thought) instructions.
        """
        return self.default_system_prompt

    def get_answer_prefix(self) -> str:
        """Returns the answer label that follows the question (e.g. 'Answer:')."""
        raise NotImplementedError

    def get_question_prompt(self, with_answer_prefill: bool = True) -> str:
        """Returns the question text.

        `with_answer_prefill=True` (the default) bakes the answer prefill into
        the returned string — required by the zero-shot / few-shot last-token
        scoring path, which reads probabilities from the very next token after
        the prefill. Set to `False` for chat-template prompting, where the
        prefill is supplied separately as the assistant turn (otherwise the
        same string ends up emitted twice and silently degrades scoring).
        """
        raise NotImplementedError

    def get_answer_from_model_output(
        self,
        last_token_probs: np.ndarray | None = None,
        tokenizer_vocab: dict[str, int] | None = None,
        text: str | None = None,
    ):
        # Token-probability decoding. Generated-text types (`GeneratedTextQA`
        # mixin) fully override this to decode from `text` instead — the two
        # decode paths are separate method overrides, not a runtime flag.
        if last_token_probs is None:
            raise ValueError("last_token_probs must be provided for token-probability QA types")
        if tokenizer_vocab is None:
            raise ValueError("tokenizer_vocab must be provided for token-probability QA types")
        return self.get_answer_from_token_probs(last_token_probs=last_token_probs, tokenizer_vocab=tokenizer_vocab)

    def get_answer_from_token_probs(
        self,
        last_token_probs: np.ndarray,
        tokenizer_vocab: dict[str, int],
    ) -> float:
        """Decodes the model's output into an answer for the given question.

        Parameters
        ----------
        last_token_probs : np.ndarray
            The model's last token probabilities for the question. The first
            dimension corresponds to the number of forward passes as specified
            by `self.num_forward_passes`.
        tokenizer : dict[str, int]
            The tokenizer's vocabulary.

        Returns
        -------
        answer : float
            The answer to the question.
        """
        raise NotImplementedError

    def get_answer_from_generated_text(
        self,
        text: str,
    ) -> float:
        """Decodes the model's outputs into an answer for the given question.

        Parameters
        ----------
        text : generated outputs
        tokenizer : dict[str, int]
            The tokenizer's vocabulary.

        Returns
        -------
        answer : float
            The answer to the question.
        """
        raise NotImplementedError

    def _hashable_params(self) -> dict:
        params = dataclasses.asdict(self)
        # `use_generated_text` is a ClassVar (absent from `asdict`) but is what
        # distinguishes a scoring type from its generated-text variant. Include
        # it so the two don't collide in task / benchmark / result identity.
        params["use_generated_text"] = self.use_generated_text
        return params

    def __hash__(self) -> int:
        return int(hash_dict(self._hashable_params()), 16)


@dataclass(frozen=True)
class DirectNumericQA(QAInterface):
    """Represents a direct numeric question.

    Notes
    -----
    For example, the prompt could be "
    Q: What is 2 + 2?
    A: "
    With the expected answer being "4".

    If looking for a direct numeric probability, the answer prompt will be
    framed as so: "
    Q: What is the probability, between 0 and 1, of getting heads on a coin flip?
    A: 0."
    So that we can extract a numeric answer with at most 2 forward passes.
    This is done automatically by passing the kwarg `answer_probability=True`.

    Note that some models have multi-digit tokens in their vocabulary, so we
    need to correctly assess which tokens in the vocabulary correspond to valid
    numeric answers.
    """

    num_forward_passes: int = 2  # NOTE: overrides superclass default
    answer_probability: bool = True

    default_system_prompt: ClassVar[str | None] = NUMERIC_SYSTEM_PROMPT
    default_chat_prompt: ClassVar[str | None] = NUMERIC_CHAT_PROMPT

    # @dataclass(frozen=True) would otherwise synthesise a field-only __hash__
    # that drops `use_generated_text`; reuse the base hash so scoring and
    # generated-text variants stay distinct.
    __hash__ = QAInterface.__hash__

    def get_answer_prefix(self) -> str:
        if self.answer_probability:
            return "Answer (between 0 and 1): 0."
        return "Answer: "

    def get_question_prompt(self, with_answer_prefill: bool = True) -> str:
        question_prompt = f"Question: {self.text}"
        if with_answer_prefill:
            question_prompt += f"\n{self.get_answer_prefix()}"

        return question_prompt

    def _get_numeric_tokens(
        self,
        tokenizer_vocab: dict[str, int],
        vocab_dim: int,
    ) -> dict[str, int]:
        """Returns the indices of tokens that correspond to numbers.

        This can include digits ("0"-"9"), multi-digit tokens (e.g., "100"), and
        the decimal point (".").

        Token ids are filtered to `< vocab_dim` (the model's logits axis); some
        tokenizer families place added/special tokens beyond the base vocab,
        and the caller indexes `last_token_probs` by these ids.

        Parameters
        ----------
        tokenizer_vocab : dict[str, int]
            The tokenizer vocabulary mapping token strings to token IDs.
        vocab_dim : int
            Size of the model's logits axis. Token IDs >= vocab_dim are excluded
            (some tokenizer families place added/special tokens beyond the base
            vocab, and the caller indexes ``last_token_probs`` by these IDs).

        Returns
        -------
        dict[str, int]
            Mapping from numeric token string to token ID, filtered to
            ``token_id < vocab_dim``.
        """
        numeric_tokens = {key: token_id for key, token_id in tokenizer_vocab.items() if key.isdigit() and token_id < vocab_dim}

        if "." in tokenizer_vocab and tokenizer_vocab["."] < vocab_dim:
            numeric_tokens["."] = tokenizer_vocab["."]

        return numeric_tokens

    def get_answer_from_token_probs(
        self,
        last_token_probs: np.ndarray,
        tokenizer_vocab: dict[str, int],
    ) -> float | int:
        """Outputs a numeric answer inferred from the model's output.

        Parameters
        ----------
        last_token_probs : np.ndarray
            The last token probabilities of the model for the question.
            The first dimension must correspond to the number for forward passes
            as specified by `num_forward_passes`.
        tokenizer_vocab: dict[str, int],
            The tokenizer's vocabulary.

        Returns
        -------
        answer : float | int
            The numeric answer to the question.

        Notes
        -----
        Eventually we could run a search algorithm to find the most likely
        answer over multiple forward passes, but for now we'll just take the
        argmax on each forward pass.
        """
        numeric_tokens_vocab = self._get_numeric_tokens(
            tokenizer_vocab,
            vocab_dim=last_token_probs.shape[-1],
        )

        if len(last_token_probs) < self.num_forward_passes:
            logging.info(f"Expected {self.num_forward_passes} forward passes, got {len(last_token_probs)}.")

        answer_text = ""
        for ltp in last_token_probs:
            # Get the probability of each numeric token
            num_tokens_probs = {
                num_token: ltp[token_id] if isinstance(ltp[token_id], float) else ltp[token_id].item()
                for num_token, token_id in numeric_tokens_vocab.items()
            }

            # Get the most likely numeric token
            most_likely_numeric_token = max(num_tokens_probs, key=lambda k: num_tokens_probs[k])
            answer_text += str(most_likely_numeric_token)

            logging.debug(f"Total prob. assigned to numeric tokens: {sum(num_tokens_probs.values()):.2%}")

        # Filter out any non-numeric characters
        match_ = re.match(r"[-+]?\d*\.\d+|\d+", answer_text)
        assert match_, f"Could not find numeric answer in '{answer_text}'."
        numeric_answer_text = match_.group()

        if self.answer_probability and "." not in numeric_answer_text:
            return float(f"0.{numeric_answer_text}")
        else:
            return float(numeric_answer_text)


@dataclass(frozen=True, eq=True)
class Choice:
    """Represents a choice in multiple-choice Q&A.

    Attributes
    ----------
    text : str
        The text of the choice. E.g., "25-34 years old".
    data_value : object
        The categorical value corresponding to this choice in the data.
    numeric_value : float, optional
        A meaningful numeric value for the choice. E.g., if the choice is "25-34
        years old", the numeric value could be 30. The choice with the highest
        numeric value can be used as a proxy for the positive class. If not
        provided, will try to use the `choice.value`.
    """

    text: str
    data_value: object
    numeric_value: float | None = None

    def get_numeric_value(self) -> float:
        """Returns the numeric value of the choice."""
        return self.numeric_value if self.numeric_value is not None else float(str(self.data_value))  # type: ignore


@dataclass(frozen=True, eq=True)  # NOTE: kw_only=True requires Python 3.10
class MultipleChoiceQA(QAInterface):
    """Represents a multiple-choice question and its answer keys."""

    num_forward_passes: int = 1  # NOTE: overrides superclass default
    choices: tuple[Choice, ...] = dataclasses.field(default_factory=tuple)
    _answer_keys_source: tuple[str, ...] = dataclasses.field(default_factory=lambda: tuple(_ALPHABET))

    def __post_init__(self):
        if not self.choices:
            raise ValueError("Choices must be provided.")
        if len(self.choices) > len(self._answer_keys_source):
            raise ValueError("Number of choices must be less than or equal to the number of answer keys.")

    def __hash__(self) -> int:
        return int(hash_dict(self._hashable_params()), 16)

    @classmethod
    def create_question_from_value_map(
        cls,
        column: str,
        value_map: dict[object, str],
        attribute: str,
        **kwargs,
    ) -> "MultipleChoiceQA":
        """Constructs a question from a value map."""
        choices = tuple(Choice(text, str(value)) for value, text in value_map.items())

        # Set default question text
        kwargs.setdefault("text", f"What is this person's {attribute}?")

        return cls(
            column=column,
            choices=choices,
            **kwargs,
        )

    @classmethod
    def create_answer_keys_permutations(cls, question: "MultipleChoiceQA") -> Iterator["MultipleChoiceQA"]:
        """Yield questions with all permutations of answer keys.

        Parameters
        ----------
        question : Question
            The template question whose answer keys will be permuted.

        Returns
        -------
        permutations : Iterator[Question]
            A generator of questions with all permutations of answer keys.
        """
        for perm in itertools.permutations(question.choices):
            yield dataclasses.replace(question, choices=perm)

    @property
    def answer_keys(self) -> tuple[str, ...]:
        return self._answer_keys_source[: len(self.choices)]

    @property
    def key_to_choice(self) -> dict[str, Choice]:
        return dict(zip(self.answer_keys, self.choices))

    @property
    def choice_to_key(self) -> dict[Choice, str]:
        return {choice: key for key, choice in self.key_to_choice.items()}

    def get_value_to_text_map(self) -> dict[object, str]:
        """Returns the map from choice data value to choice textual representation."""
        return {choice.data_value: choice.text for choice in self.choices}

    def get_answer_key_from_value(self, value: object) -> str | None:
        """Returns the answer key corresponding to the given data value."""
        for choice in self.choices:
            if choice.data_value == value:
                return self.choice_to_key[choice]

        logging.error(f"Could not find choice for value: {value}")
        return None

    def get_choice_from_answer_key(self, text: str) -> Choice | None:
        """Returns the choice object corresponding to the answer key (letter)."""
        text = text.strip().upper()
        if text in self.key_to_choice:
            return self.key_to_choice[text]

        logging.error(f"Could not find answer choice for text: {text}")
        return None

    def get_answer_prefix(self) -> str:
        return "Answer:"

    def get_question_prompt(self, with_answer_prefill: bool = True) -> str:
        choice_str = "\n".join(f"{key}. {choice.text}." for key, choice in self.key_to_choice.items())

        prompt = f"Question: {self.text}\n{choice_str}"
        if with_answer_prefill:
            prompt += f"\n{self.get_answer_prefix()}"
        return prompt

    def _decode_model_output_to_choice_distribution(
        self,
        last_token_probs: np.ndarray,
        tokenizer_vocab: dict[str, int],
    ) -> dict[Choice, float]:
        """Decodes the model's output into an answer distribution.

        Parameters
        ----------
        last_token_probs : np.ndarray
            The model's last token probabilities for the question.
        tokenizer_vocab: dict[str, int],
            The tokenizer's vocabulary.

        Returns
        -------
        answers : dict[Choice, float]
            How much probability the model places on each answer choice.

        Notes
        -----
        Answer-key tokens may be prefixed with a space, so we need to check
        both "A" and " A" templates.
        """

        def _get_choice_token_id(choice: Choice, prefix: str = " ") -> int | None:
            choice_answer_text = f"{prefix}{self.choice_to_key[choice]}"
            if choice_answer_text in tokenizer_vocab:
                return tokenizer_vocab[choice_answer_text]
            else:
                return None

        # Different models may use different prefixes to represent white space
        # or word boundaries; here we try a few common ones
        prefixes = ["", " ", "_", "\u2581", "\u0120", "\u010a"]

        # Map probabilities to choice values
        answers_per_prefix = {
            prf: {
                choice: last_token_probs[choice_token_id].item()
                for choice in self.choices
                if (choice_token_id := _get_choice_token_id(choice, prefix=prf)) is not None
            }
            for prf in prefixes
        }

        # Choose the prefix with the highest probability density
        best_prefix = max(answers_per_prefix, key=lambda prf: sum(answers_per_prefix[prf].values()))
        answers = answers_per_prefix[best_prefix]

        # Log prefix information in debug mode
        for prefix, choice_probs in answers_per_prefix.items():
            logging.debug(f"prefix='{prefix}' has density {sum(choice_probs.values()):.2%}")

        # Normalize probabilities to sum to 1
        answers_sum_prob = sum(answers.values())

        # Log total probability density assigned to answers
        msg = f"Answers have {answers_sum_prob:.2%} probability assigned."
        if answers_sum_prob < ANSWER_PROB_THRESHOLD:
            id_to_tok = {v: k for k, v in tokenizer_vocab.items()}
            argmax_id = int(np.argmax(last_token_probs))
            argmax_token = id_to_tok.get(argmax_id, f"<id={argmax_id}>")
            logging.warning(msg + f" Argmax token: '{argmax_token}'.")
        else:
            logging.debug(msg)

        # No mass on any choice token — happens when the top-K logprobs cap
        # excludes all answer-letter variants (vLLM/WebAPI), or with extreme
        # FP16 underflow on transformers. Fall back to uniform over the QA's
        # declared choices: same effect as the model saying "I don't know."
        if answers_sum_prob <= 0 or not answers:
            n = len(self.choices)
            return {choice: 1.0 / n for choice in self.choices}

        return {choice: prob / answers_sum_prob for choice, prob in answers.items()}

    def get_answer_from_token_probs(
        self,
        last_token_probs: np.ndarray,
        tokenizer_vocab: dict[str, int],
    ) -> float:
        """Decodes the model's output into an answer for the given question.

        Parameters
        ----------
        last_token_probs : np.ndarray
            The model's last token probabilities for the question. The first
            dimension corresponds to the number of forward passes as specified
            by `self.num_forward_passes`.
        tokenizer_vocab: dict[str, int],
            The tokenizer's vocabulary.

        Returns
        -------
        answer : float
            The answer to the question.
        """
        if last_token_probs.ndim > 1:
            if last_token_probs.shape[0] > 1:
                logging.warning(f"Multiple ({last_token_probs.shape[0]}) forward passes detected: using only the first pass.")

            # Using only 1st forward pass results
            last_token_probs = last_token_probs[0]

        answers = self._decode_model_output_to_choice_distribution(
            last_token_probs=last_token_probs,
            tokenizer_vocab=tokenizer_vocab,
        )

        sorted_choices_by_value = sorted(
            answers.keys(),
            key=lambda choice: choice.get_numeric_value(),
        )

        # If binary question, return probability of positive answer
        # > positive answer always has the highest numeric value
        if len(answers) == 2:
            positive_choice = sorted_choices_by_value[-1]
            return answers[positive_choice]

        # Compute risk estimate by summing weighted choices
        risk_estimate = sum(choice.get_numeric_value() * prob for choice, prob in answers.items())

        logging.debug(f"Risk estimate: {risk_estimate:.2f}")
        return risk_estimate


# At runtime `GeneratedTextQA` is a plain mixin (combined with a real
# `QAInterface` subclass). For type-checking we declare it as a `QAInterface`
# so `super()` / `self` calls to the base interface (get_question_prompt,
# get_answer_from_generated_text, …) type-check.
if TYPE_CHECKING:
    _GeneratedTextQABase = QAInterface
else:
    _GeneratedTextQABase = object


class GeneratedTextQA(_GeneratedTextQABase):
    """Mixin for QA types that extract the answer from generated text.

    Combined with a base answer type (`DirectNumericQA` / `MultipleChoiceQA`),
    it flips the *decoding* axis from token-probability scoring to free-form
    text generation + regex extraction, while the base type keeps owning the
    *answer semantics* (numeric value / choices). It is a plain mixin (no
    dataclass fields of its own) mixed in first so its ClassVars/methods win:
    `class TextNumericQA(GeneratedTextQA, DirectNumericQA)`.

    The three differences from the scoring path, in one place:
    - ``use_generated_text = True``
    - ``default_chat_prompt = None`` — no assistant prefill. The prefill (e.g.
      ``"Answer (between 0 and 1): 0."``) is a *scoring* construct; on the
      generation path it would fight the ``format_instruction`` the model is
      told to follow. This is what removes the chat-prefill contradiction.
    - the question prompt omits the answer prefill, and answer decoding routes
      to ``get_answer_from_generated_text`` (validated to receive ``text``).
    """

    use_generated_text: ClassVar[bool] = True
    default_chat_prompt: ClassVar[str | None] = None
    cot_instruction: ClassVar[str] = COT_INSTRUCTION
    max_new_tokens: int = 8000

    # Answer-format instruction appended to the default system prompt so the
    # model emits parser-friendly output (see `get_default_system_prompt`).
    # `None` = no instruction; overridden per concrete text type.
    format_instruction: ClassVar[str | None] = None

    # Whether to prepend the chain-of-thought instruction to the system prompt
    # (before the answer-format instruction). Declared here as a plain instance
    # annotation, so it can be overwritten per-instance and part of QA identity.
    use_cot: bool = False

    def get_default_system_prompt(self) -> str | None:
        """System prompt for the generated-text path.

        Starts from the type's ``default_system_prompt`` and appends, in order,
        the optional ``cot_instruction`` (when ``use_cot`` is set — reason first)
        and the ``format_instruction`` (so the model emits parser-friendly
        output last).
        """
        base = self.default_system_prompt
        if base is None:
            return None

        parts = [base.rstrip()]
        if self.use_cot and self.cot_instruction:
            parts.append(self.cot_instruction)
        if self.format_instruction:
            parts.append(self.format_instruction)
        return "\n".join(parts)

    def get_answer_prefix(self) -> str:
        # Generated-text QA has no scoring prefill (the answer is parsed from
        # free-form output). Return empty so no prefill leaks in via callers
        # that emit the prefix directly, e.g. `VarySuffix` with show_question=False.
        return ""

    def get_question_prompt(self, with_answer_prefill: bool = True) -> str:  # noqa: D102
        # Generation never bakes in an answer prefill (the model must produce
        # free-form text we then parse); ignore `with_answer_prefill`.
        return super().get_question_prompt(with_answer_prefill=False)

    def get_answer_from_model_output(
        self,
        last_token_probs: np.ndarray | None = None,
        tokenizer_vocab: dict[str, int] | None = None,
        text: str | None = None,
    ):
        if text is None:
            raise ValueError("text must be provided for generated-text QA types")
        return self.get_answer_from_generated_text(text=text)

    @classmethod
    def from_base(cls, base_qa: QAInterface, *, use_cot: bool = False) -> "MultipleChoiceQA | DirectNumericQA":
        """Build the generated-text variant of a base QA instance.

        Returns a `TextNumericQA` / `TextMultipleChoiceQA`, which are subclasses
        of `DirectNumericQA` / `MultipleChoiceQA` (hence the return type).

        `dataclasses.replace` cannot change an object's class, so switching a
        task between scoring and generation decoding (see
        `TaskMetadata.use_text_output_for_qa`) rebuilds the question as the
        matching generated-text type, carrying over all its fields.

        `use_cot` toggles the chain-of-thought instruction in the system prompt.
        """
        text_cls = _BASE_TO_TEXT_QA.get(type(base_qa))
        if text_cls is None:
            raise TypeError(f"No generated-text QA type registered for {type(base_qa).__name__}.")
        field_values = {f.name: getattr(base_qa, f.name) for f in dataclasses.fields(base_qa)}
        return text_cls(**field_values, use_cot=use_cot)


@dataclass(frozen=True)
class TextNumericQA(GeneratedTextQA, DirectNumericQA):
    """Numeric QA decoded from generated text (regex), not token probabilities."""

    use_cot: bool = False
    format_instruction: ClassVar[str | None] = NUMERIC_GENERATED_TEXT_FORMAT
    __hash__ = QAInterface.__hash__  # keep distinct from DirectNumericQA (see _hashable_params)

    def get_answer_from_generated_text(self, text: str) -> float:
        """Extract the probability answer from the model's generated text.

        Parameters
        ----------
        text : str
            The model's generated output for the question.

        Returns
        -------
        answer : float
            The probability in [0, 1], or ``np.nan`` when nothing could be parsed
            (so the failure is distinguishable from a genuine 0.5; the caller
            drops NaNs and/or counts them as extraction failures).
        """
        probability = self.extract_probability_from_text(text)

        if probability is None:
            logging.warning(f"No probability found in generated text: {text!r}; returning NaN.")
            return float(np.nan)

        logging.debug(f"Extracted probability: {probability:.2%}")
        return probability

    @staticmethod
    def extract_probability_from_text(generated_text: str) -> float | None:
        """Extract a probability value from generated text using regex patterns.

        The extraction prioritizes (in order): the explicit "Probability: X[%]"
        anchor, last loose percentage, "X percent", then a bare 0.XX decimal.
        Returns a float in [0, 1] or None if nothing matched.
        """
        # First, try the explicit "Probability: X[%]" anchor (most reliable).
        explicit_patterns = [
            (
                r"[Pp]robability(?:\s+(?:is|of|estimate)?)?[:\s]+(\d+(?:\.\d+)?)\s*%",
                True,
            ),
            (
                r"[Pp]robability(?:\s+(?:is|of|estimate)?)?[:\s]+(\d*\.?\d+)(?![%\d])",
                False,
            ),
        ]
        for pattern, percent_form in explicit_patterns:
            value = TextNumericQA._extract_last_probability(
                generated_text,
                pattern,
                percent_form=percent_form,
            )
            if value is not None:
                logging.debug(f"Extracted probability {value:.2%} using pattern: {pattern}")
                return value

        # Fallback ladder: any percentage, "X percent", or a bare 0.XX decimal.
        # `flags=0` for the percent forms; "percent" pattern is case-insensitive.
        for pattern, percent_form, flags in [
            (r"(\d+(?:\.\d+)?)\s*%", True, 0),
            (r"(\d+(?:\.\d+)?)\s+percent", True, re.IGNORECASE),
            (r"(?<![.\d])(0?\.\d+)(?![.\d])", False, 0),
        ]:
            value = TextNumericQA._extract_last_probability(
                generated_text,
                pattern,
                percent_form=percent_form,
                flags=flags,
            )
            if value is not None:
                logging.debug(f"Used fallback extraction: {value:.2%}")
                return value

        snippet = generated_text[:250] + "..." + generated_text[-250:] if len(generated_text) > 500 else generated_text
        logging.error(f"Could not extract probability from text:\n{snippet}")
        return None

    @staticmethod
    def _extract_last_probability(
        text: str,
        pattern: str,
        *,
        percent_form: bool,
        flags: int = 0,
    ) -> float | None:
        """Apply `pattern` to `text`, take the last match, and return it as a
        probability in [0, 1] (dividing by 100 if `percent_form`) or None.

        The "last match" rule matters: models often revise their estimate
        mid-reasoning, and the final value is the one we want to trust.
        """
        matches = re.findall(pattern, text, flags=flags)
        if not matches:
            return None
        value = float(matches[-1])
        # If the pattern is the explicit `Probability: <number>(?!%)` form,
        # callers may still emit a value > 1 they meant as a percentage.
        if value > 1:
            value = value / 100.0
        elif percent_form:
            value = value / 100.0
        if 0 <= value <= 1:
            return value
        logging.warning(f"Extracted value {value} is out of range [0, 1]")
        return None


@dataclass(frozen=True)
class TextMultipleChoiceQA(GeneratedTextQA, MultipleChoiceQA):
    """Multiple-choice QA decoded from generated text (regex), not token probabilities.

    Inherits the choice/answer-key machinery from `MultipleChoiceQA` and adds
    the text-extraction path (parse an answer key from free-form output, then
    map it to the corresponding choice).
    """

    use_cot: bool = False
    format_instruction: ClassVar[str | None] = MCQ_GENERATED_TEXT_FORMAT
    __hash__ = QAInterface.__hash__  # keep distinct from MultipleChoiceQA (see _hashable_params)

    def get_answer_from_generated_text(self, text: str) -> float:
        """Decodes the model's generated text into a risk estimate.

        Parameters
        ----------
        text : str
            The model's generated output for the question.

        Returns
        -------
        answer : float
            The risk estimate, or ``np.nan`` when no answer can be parsed (so the
            failure is distinguishable from a genuine 0.5).
        """
        choices = self._decode_generated_text_to_choice_distribution(text=text)
        if not choices:
            return float(np.nan)

        sorted_choices_by_value = sorted(
            choices.keys(),
            key=lambda choice: choice.get_numeric_value(),
        )

        # If binary question, return probability of positive answer
        # > positive answer always has the highest numeric value
        if len(choices) == 2:
            positive_choice = sorted_choices_by_value[-1]
            return choices[positive_choice]

        # Compute risk estimate by summing weighted choices
        risk_estimate = sum(choice.get_numeric_value() * prob for choice, prob in choices.items())

        logging.debug(f"Risk estimate: {risk_estimate:.2f}")
        return risk_estimate

    def extract_answer_key_from_text(self, generated_text: str) -> str | None:
        """Extract the answer key from a model output using regex patterns.

        Mirrors `TextNumericQA.extract_probability_from_text`: first try the
        explicit "Answer:"-anchored keys (most reliable), then fall back to a
        looser, unanchored ladder. Returns the answer key or None if nothing
        matched.
        """
        # First, try keys anchored to an explicit "Answer:" indicator.
        key = self._extract_last_answer_key(generated_text, anchored=True)
        if key is not None:
            logging.debug(f"Extracted answer key {key!r} from explicit 'Answer:' anchor")
            return key

        # Fallback: looser, unanchored key/choice-text matches.
        key = self._extract_last_answer_key(generated_text, anchored=False)
        if key is not None:
            logging.debug(f"Extracted answer key {key!r} from fallback ladder")
            return key

        # Last resort: the model replied with just the bare key (e.g. "B"). This
        # is excluded from the unanchored ladder (a lone "A" collides with the
        # article), but it's unambiguous when the *entire* response is the key.
        stripped = generated_text.strip()
        for key in self.answer_keys:
            if stripped.casefold() == key.casefold():
                logging.debug(f"Extracted answer key {key!r} as the whole response")
                return key

        logging.error(f"No answer found in text: {generated_text}")
        return None

    def _extract_last_answer_key(self, text: str, *, anchored: bool) -> str | None:
        """Scan `text` for an answer key, most-specific pattern tier first.

        Within the first tier that matches any key, the LAST occurrence wins
        (models often revise mid-reasoning, so the final mention is the one to
        trust) — matching `TextNumericQA._extract_last_probability`.

        When `anchored`, every pattern must be preceded by an "Answer:"
        indicator. The bare-key tier (a lone letter like "A") is only used in
        the anchored pass: unanchored, a single letter collides with the English
        word/article "A".
        """
        answer_prefix = f"(?:{'|'.join(_ANSWER_PATTERNS)})" if anchored else ""
        no_alphanumeric_before = "(?<![A-Za-z0-9])"
        no_alphanumeric_after = "(?![A-Za-z0-9])"
        optional_punct_ws = r"[\.\)\-:]?\s*"

        # Pattern tiers, most specific first (index 0 = highest priority).
        def tiers_for(key: str, choice_text: str) -> list[str]:
            tiers = [
                # key + punctuation + choice text (e.g. "A) Answer Text")
                rf"{answer_prefix}{no_alphanumeric_before}{key}{no_alphanumeric_after}{optional_punct_ws}{choice_text}",
                # key + punctuation only (e.g. "A." or "B)")
                rf"{answer_prefix}{no_alphanumeric_before}{key}{no_alphanumeric_after}[\.\)\-:]",
                # choice text only (e.g. "Answer Text"), bounded so short choice
                # words ("No"/"Yes") don't match inside other words ("nothing",
                # "cannot"). A bare choice word that IS the whole word (e.g. "no"
                # in "no answer") remains inherently ambiguous — the anchored
                # "Answer: X" letter format is the reliable path.
                rf"{answer_prefix}{no_alphanumeric_before}(?:{choice_text}|{choice_text.lower()}){no_alphanumeric_after}",
            ]
            if anchored:
                # bare key (e.g. "Answer: A") -- only safe behind the anchor
                tiers.append(rf"{answer_prefix}{no_alphanumeric_before}{key}{no_alphanumeric_after}")
            return tiers

        # Precompute per-key patterns once (choices are instance-specific).
        per_key_patterns: dict[str, list[str]] = {}
        for key in self.answer_keys:
            choice = self.get_choice_from_answer_key(key)
            if choice is None:
                continue
            per_key_patterns[key] = tiers_for(key, re.escape(choice.text))

        if not per_key_patterns:
            return None

        # Tier-major so a more specific tier always beats a looser one; within a
        # tier, the last-positioned match across keys wins.
        n_tiers = len(next(iter(per_key_patterns.values())))
        for tier in range(n_tiers):
            best_key, best_pos = None, -1
            for key, patterns in per_key_patterns.items():
                matches = list(re.finditer(patterns[tier], text))
                if not matches:
                    continue
                pos = matches[-1].start()
                if pos > best_pos:
                    best_pos, best_key = pos, key
            if best_key is not None:
                return best_key
        return None

    def _decode_generated_text_to_choice_distribution(
        self,
        text: str,
    ) -> dict[Choice, float]:
        # Degenerate distribution (mass 1 on the identified choice) for a single
        # text answer. Returns an empty dict when no answer can be parsed, so the
        # caller can signal the failure (NaN) instead of a uniform 0.5.
        answer_key = self.extract_answer_key_from_text(generated_text=text)
        if answer_key is None:
            logging.warning(f"No answer key parsed from generated text: {text!r}; returning empty distribution.")
            return {}
        choice = self.get_choice_from_answer_key(answer_key)
        if choice is None:
            logging.warning(f"Parsed answer key {answer_key!r} maps to no choice; returning empty distribution.")
            return {}
        # A single text answer yields a hard label, so the distribution is
        # degenerate (mass 1 on the chosen option, 0 elsewhere) — the resulting
        # risk estimate is 0/1, not a calibrated probability.
        logging.debug(
            f"Extracted answer '{choice.text}'; single generated text gives a "
            f"degenerate 0/1 distribution (not a calibrated probability)."
        )
        return {c: float(c == choice) for c in self.choices}


# Maps each scoring QA type to its generated-text counterpart (used by
# `GeneratedTextQA.from_base`). Defined after the subclasses exist.
_BASE_TO_TEXT_QA: dict[type, type] = {
    DirectNumericQA: TextNumericQA,
    MultipleChoiceQA: TextMultipleChoiceQA,
}
