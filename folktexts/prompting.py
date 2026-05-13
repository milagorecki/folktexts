"""Module for prompt construction and variation.

A prompt for a tabular row has three parts:
  [PREFIX]  task description / system context  (same for all rows)
  [INFO]    serialized feature-value pairs     (row-specific)
  [SUFFIX]  question text

Within INFO the pipeline is fixed by semantics:
  VaryValueMap → VaryOrder → VaryConnector → VaryFormat

Return types enforce the order: per-item stages share list→list; VaryFormat
collapses the list to str, making it impossible to apply a per-item stage after it.

Allows for zero-shot, few-shot, and chat-template prompting styles.
"""

from __future__ import annotations

import dataclasses
import logging
from copy import copy
from dataclasses import dataclass, field
from string import Template
from typing import Any, ClassVar

import pandas as pd
from transformers import AutoTokenizer

from folktexts.acs import ACS_TASK_DESCRIPTION, ACS_TASK_DESCRIPTION_DEFAULTS
from folktexts.sipp import SIPP_TASK_DESCRIPTION, SIPP_TASK_DESCRIPTION_DEFAULTS

try:
    from folktexts.ts import TABLESHIFT_TASK_DESCRIPTION, TABLESHIFT_TASK_DESCRIPTION_DEFAULTS
except ImportError:
    TABLESHIFT_TASK_DESCRIPTION = Template("")
    TABLESHIFT_TASK_DESCRIPTION_DEFAULTS = dict()

from .dataset import Dataset
from .qa_interface import MultipleChoiceQA, QAInterface
from .task import TaskMetadata

# Sentinel distinguishing "use the mode-appropriate default" from `None`
# ("explicitly disable the role"). Module-private; not part of the public API.
_DEFAULT = object()

SYSTEM_PROMPT = """\
You are a helpful assistant. You answer multiple-choice questions \
based on the information provided. Respond with a single answer choice.
"""

NUMERIC_SYSTEM_PROMPT = """\
You are a helpful assistant. You provide numeric probability \
estimates based on the information provided.
"""

ANTHROPIC_CHAT_PROMPT = """If had to select one of the options, my answer would be"""
GEMMA_CHAT_PROMPT = """The provided information suggests that the answer is"""
# NOTE: The leading `0.` is part of the prefill, so the model only generates
# the digits after the decimal point. This caps the expressible probability
# at the open interval [0, 1) — true posteriors at or near 1.0 cannot be
# emitted exactly. If you need full [0, 1] coverage, override `chat_prompt`
# with e.g. `"Answer (between 0 and 1): "` and let the model produce the
# leading digit itself (note that this also widens the digit-scoring search
# space and may degrade calibration for low-probability cases).
NUMERIC_CHAT_PROMPT = """Answer (between 0 and 1): 0."""

DEFAULT_PROMPT_STYLE = {
    "format": "textbullet",
    "connector": "is",
    "granularity": "original",
    "order": None,
    "custom_prompt_prefix": None,
    "custom_prompt_suffix": None,
    "show_question": True,
}


# ---------------------------------------------------------------------------
# Intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class FeatureItem:
    col: str  # pandas column name
    label: str  # human-readable name from ColumnToText.short_description
    raw_value: Any  # original value from the DataFrame
    text_value: str = ""  # filled by VaryValueMap
    connected: str = ""  # filled by VaryConnector


# ---------------------------------------------------------------------------
# Variation stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VaryPrefix:
    task_description: str
    add_task_description: bool = True
    custom_prefix: str | None = None

    def __call__(self) -> str:
        if not self.add_task_description:
            # custom_prefix is intentionally ignored when task description is suppressed
            return "Information:\n"
        if self.custom_prefix:
            cp = self.custom_prefix if self.custom_prefix.endswith("\n") else self.custom_prefix + "\n"
            return self.task_description + cp + "\nInformation:\n"
        return self.task_description + "\nInformation:\n"


@dataclass(frozen=True)
class VarySuffix:
    question: QAInterface
    show_question: bool = True
    show_label: bool = False
    label: Any = None  # only used when show_label=True
    custom_suffix: str | None = None

    def __call__(self) -> str:
        base = self.question.get_question_prompt() if self.show_question else self.question.get_answer_prefix()
        label_part = f" {self.label}\n\n" if self.show_label else ""
        return f"\n{base}{label_part}{self.custom_suffix or ''}"


@dataclass(frozen=True)
class VaryValueMap:
    cols_to_text: dict = field(hash=False, compare=False)

    def __call__(self, items: list[FeatureItem]) -> list[FeatureItem]:
        return [dataclasses.replace(item, text_value=self.cols_to_text[item.col][item.raw_value]) for item in items]

    @classmethod
    def with_low_granularity(cls, cols_to_text: dict, simplified_value_maps: dict) -> "VaryValueMap":
        """Return a VaryValueMap with simplified (low-granularity) value maps.

        Shallow-copies each ColumnToText that has a simplified map available,
        leaving the original task object untouched.
        """
        modified = {}
        for col, c2t in cols_to_text.items():
            if col in simplified_value_maps:
                c2t_copy = copy(c2t)
                c2t_copy._value_map = simplified_value_maps[col]
                modified[col] = c2t_copy
            else:
                modified[col] = c2t
        return cls(cols_to_text=modified)


@dataclass(frozen=True)
class VaryOrder:
    order: list | None = None  # list[str] of column names; None → keep original

    def __call__(self, items: list[FeatureItem]) -> list[FeatureItem]:
        if not self.order:
            return items
        index = {item.col: item for item in items}
        return [index[col] for col in self.order if col in index]


VaryFeatureOrder = VaryOrder  # alias for backward compatibility


@dataclass(frozen=True)
class VaryConnector:
    connector: str = "is"

    def __call__(self, items: list[FeatureItem]) -> list[FeatureItem]:
        sep = ": " if self.connector == ":" else f" {self.connector} "
        return [dataclasses.replace(item, connected=f"{item.label}{sep}{item.text_value}") for item in items]


@dataclass(frozen=True)
class VaryFormat:
    format: str = "textbullet"

    _TEMPLATES: ClassVar[dict] = {
        "bullet": lambda s: f"- {s}\n",
        "comma": lambda s: f"{s}, ",
        "text": lambda s: f"The {s}. ",
        "textbullet": lambda s: f"- The {s}.\n",
    }

    def __post_init__(self):
        if self.format not in self._TEMPLATES:
            raise ValueError(f"Unknown format {self.format!r}. Choose from {list(self._TEMPLATES)}")

    def __call__(self, items: list[FeatureItem]) -> str:
        template = self._TEMPLATES[self.format]
        return "".join(template(item.connected) for item in items)


@dataclass(frozen=True)
class VarySystemPrompt:
    system_prompt: str

    def __call__(self) -> str:
        return self.system_prompt


# ---------------------------------------------------------------------------
# PromptConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptConfig:
    prefix: VaryPrefix
    value_map: VaryValueMap
    order: VaryOrder
    connector: VaryConnector
    format: VaryFormat
    suffix: VarySuffix
    system_prompt: VarySystemPrompt | None = None

    @classmethod
    def default(cls, task: TaskMetadata) -> "PromptConfig":
        return _build_config(task)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(
    task: TaskMetadata,
    question: QAInterface | None = None,
    add_task_description: bool = True,
    custom_prompt_prefix: str | None = None,
    custom_prompt_suffix: str | None = None,
    prompt_variation: dict | None = None,
    system_prompt: str | None = None,
) -> PromptConfig:
    pv = prompt_variation or {}
    question = question or task.question

    granularity = pv.get("granularity", DEFAULT_PROMPT_STYLE["granularity"])
    value_map = (
        VaryValueMap.with_low_granularity(task.cols_to_text, _get_simplified_value_maps(task))
        if granularity == "low"
        else VaryValueMap(task.cols_to_text)
    )

    order_raw = pv.get("order", DEFAULT_PROMPT_STYLE["order"])
    if isinstance(order_raw, str):
        order_raw = [col.strip() for col in order_raw.split(",")]

    return PromptConfig(
        prefix=VaryPrefix(
            task_description=_get_task_description(task, pv.get("task_description")),
            add_task_description=add_task_description,
            custom_prefix=pv.get("custom_prompt_prefix", custom_prompt_prefix),
        ),
        value_map=value_map,
        order=VaryOrder(order=order_raw),
        connector=VaryConnector(connector=pv.get("connector", DEFAULT_PROMPT_STYLE["connector"])),
        format=VaryFormat(format=pv.get("format", DEFAULT_PROMPT_STYLE["format"])),
        suffix=VarySuffix(
            question=question,
            show_question=pv.get("show_question", DEFAULT_PROMPT_STYLE["show_question"]),
            custom_suffix=pv.get("custom_prompt_suffix", custom_prompt_suffix),
        ),
        system_prompt=VarySystemPrompt(system_prompt) if system_prompt is not None else None,
    )


def _get_task_description(task: TaskMetadata, override: str | None = None) -> str:
    if override is not None:
        return override
    descriptions = {
        "ACS": ACS_TASK_DESCRIPTION.substitute(ACS_TASK_DESCRIPTION_DEFAULTS),
        "SIPP": SIPP_TASK_DESCRIPTION.substitute(SIPP_TASK_DESCRIPTION_DEFAULTS),
    }
    if TABLESHIFT_TASK_DESCRIPTION is not None:
        descriptions["BRFSS"] = TABLESHIFT_TASK_DESCRIPTION.substitute(TABLESHIFT_TASK_DESCRIPTION_DEFAULTS)
    for key, desc in descriptions.items():
        if key in task.name:
            return desc
    raise ValueError(f"Cannot determine task description for task '{task.name}'")


def _get_simplified_value_maps(task: TaskMetadata) -> dict:
    if task.name.startswith("ACS"):
        from folktexts.acs.acs_columns_alt import simplified_value_maps

        return simplified_value_maps
    raise NotImplementedError(f"Low-granularity value maps are not available for task '{task.name}'.")


def _get_few_shot_task_description(task: TaskMetadata) -> str | None:
    overrides = {
        "respondent": "different survey respondents",
        "suffix": " for each person",
    }
    if task.name.startswith("ACS"):
        return ACS_TASK_DESCRIPTION.substitute({**ACS_TASK_DESCRIPTION_DEFAULTS, **overrides})
    if TABLESHIFT_TASK_DESCRIPTION is not None and "BRFSS" in task.name:
        return TABLESHIFT_TASK_DESCRIPTION.substitute({**TABLESHIFT_TASK_DESCRIPTION_DEFAULTS, **overrides})
    if task.name.startswith("SIPP"):
        return SIPP_TASK_DESCRIPTION.substitute({**SIPP_TASK_DESCRIPTION_DEFAULTS, **overrides})
    return None


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    def __init__(self, task: TaskMetadata):
        self.task = task

    def _extract_items(self, row: pd.Series) -> list[FeatureItem]:
        return [
            FeatureItem(
                col=col,
                label=self.task.cols_to_text[col].short_description,
                raw_value=row[col],
            )
            for col in self.task.features
            if col in row.index
        ]

    def build(self, row: pd.Series, config: PromptConfig) -> str:
        items = self._extract_items(row)
        items = config.value_map(items)
        items = config.order(items)
        items = config.connector(items)
        info_block = config.format(items)
        if not info_block.endswith("\n"):
            info_block = info_block.rstrip(", ") + "\n"
        return config.prefix() + info_block + config.suffix()

    def build_few_shot(
        self,
        row: pd.Series,
        config: PromptConfig,
        examples: list[tuple],  # list of (pd.Series, label)
        few_shot_task_description: str | None = None,
        example_order: list[int] | None = None,
    ) -> str:
        if example_order is not None:
            assert len(example_order) == len(examples)
            examples = [examples[i] for i in example_order]

        parts = []
        for i, (ex_row, ex_label) in enumerate(examples):
            if i == 0:
                prefix = config.prefix
                if few_shot_task_description is not None:
                    prefix = dataclasses.replace(prefix, task_description=few_shot_task_description)
            else:
                prefix = dataclasses.replace(config.prefix, add_task_description=False)
            ex_config = dataclasses.replace(
                config,
                prefix=prefix,
                suffix=dataclasses.replace(
                    config.suffix,
                    show_question=False,
                    show_label=True,
                    label=ex_label,
                ),
            )
            parts.append(self.build(ex_row, ex_config))

        target_config = dataclasses.replace(
            config,
            prefix=dataclasses.replace(config.prefix, add_task_description=False),
        )
        parts.append(self.build(row, target_config))
        return "".join(parts)

    def build_chat(
        self,
        row: pd.Series,
        config: PromptConfig,
        tokenizer: AutoTokenizer,
        chat_prompt: str | None = None,
        **kwargs,
    ) -> str:
        user_content = self.build(row, config)
        system_content = config.system_prompt() if config.system_prompt else None
        return apply_chat_template(
            tokenizer,
            user_prompt=user_content,
            system_prompt=system_content,
            chat_prompt=chat_prompt,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def encode_row_prompt(
    row: pd.Series,
    task: TaskMetadata,
    question: QAInterface = None,
    custom_prompt_prefix: str = None,
    custom_prompt_suffix: str = None,
    add_task_description: bool = True,
    prompt_variation: dict | None = None,
) -> str:
    """Encode a question regarding a given row into a natural-language prompt.

    Parameters
    ----------
    row : pd.Series
        The data row to encode.
    task : TaskMetadata
        The task that defines features, column mappings, and the question.
    question : QAInterface, optional
        The question interface to use; defaults to ``task.question``.
    custom_prompt_prefix : str, optional
        Text to prepend before the task description.
    custom_prompt_suffix : str, optional
        Text to append after the row encoding (before the question).
    add_task_description : bool, optional
        Whether to include the task description in the prefix, by default True.
    prompt_variation : dict | None, optional
        Prompt style overrides. Supported keys:

        - ``"format"`` : str, default ``"textbullet"`` — row serialization
          format; one of ``"bullet"``, ``"text"``, ``"textbullet"``, ``"comma"``.
        - ``"connector"`` : str, default ``"is"`` — verb linking feature name
          to value; one of ``"is"``, ``"="``, ``":"``.
        - ``"granularity"`` : str, default ``"original"`` — value-map
          granularity; one of ``"original"``, ``"low"``.
        - ``"order"`` : list[str] | None, default ``None`` — column names in
          the desired display order; ``None`` keeps the task's default order.
        - ``"custom_prompt_prefix"`` : str | None — alternative to the
          ``custom_prompt_prefix`` parameter above.
        - ``"custom_prompt_suffix"`` : str | None — alternative to the
          ``custom_prompt_suffix`` parameter above.
        - ``"show_question"`` : bool, default ``True`` — if ``False``, omit
          the question text and emit only the answer prefix.
        - ``"example_order"`` : list[int] | None — few-shot only; reorders
          examples by index before building the prompt.

    Returns
    -------
    str
        The fully formatted prompt string.
    """
    row = row[task.features]
    config = _build_config(
        task=task,
        question=question,
        add_task_description=add_task_description,
        custom_prompt_prefix=custom_prompt_prefix,
        custom_prompt_suffix=custom_prompt_suffix,
        prompt_variation=prompt_variation,
    )
    return PromptBuilder(task).build(row, config)


def encode_row_prompt_few_shot(
    row: pd.Series,
    task: TaskMetadata,
    dataset: Dataset,
    n_shots: int,
    question: QAInterface = None,
    reuse_examples: bool = False,
    compose_few_shot_examples: str | list = "random",
    example_order: list[int] | None = None,
    custom_prompt_prefix: str = None,
    prompt_variation: dict | None = None,
) -> str:
    """Encode a question regarding a given row using few-shot prompting.

    Parameters
    ----------
    row : pd.Series
        The row that the question will be about.
    task : TaskMetadata
        The task that the row belongs to.
    dataset : Dataset
        The dataset to draw few-shot examples from (sampled from the train split).
    n_shots : int
        The number of example questions and answers to prepend.
    question : QAInterface, optional
        The question interface to use; defaults to ``task.question``.
    reuse_examples : bool, optional
        Whether to reuse the same examples for consistency. By default will
        resample new examples each time (`reuse_examples=False`).
    compose_few_shot_examples : str or list, optional
        How to select few-shot samples: ``"random"`` (default), ``"balanced"``
        (equal draws per class), or a list of per-class counts summing to
        ``n_shots``.
    example_order : list[int] | None, optional
        Integer permutation to reorder examples before building the prompt
        (e.g. ``[2, 0, 1]`` for 3 shots). ``None`` keeps the sampled order.
    custom_prompt_prefix : str, optional
        A custom prefix to prepend before the task description.
    prompt_variation : dict | None, optional
        Prompt style overrides (format, connector, granularity, order, etc.).

    Returns
    -------
    prompt : str
        The encoded few-shot prompt.
    """
    logging.debug(f"Composition of few shot examples: {compose_few_shot_examples}")

    # Take `n_shots` random samples from the train set
    X_examples, y_examples = dataset.sample_n_train_examples(
        n_shots,
        reuse_examples=reuse_examples,
        composition=compose_few_shot_examples,
    )
    X_examples = X_examples.sort_index()
    y_examples = y_examples.sort_index()
    logging.debug(f"ys index: {y_examples.index.tolist()}")
    logging.debug(f"ys: {y_examples.values.tolist()}")

    # Get the question to ask
    question = question or task.question

    pv = dict(prompt_variation) if prompt_variation else {}
    if not example_order:
        example_order_raw = pv.pop("example_order", None)
        if isinstance(example_order_raw, str):
            example_order = [int(i) for i in example_order_raw.split(",")]
        else:
            example_order = example_order_raw

    examples = []
    for i in range(n_shots):
        label = (
            question.get_answer_key_from_value(y_examples.iloc[i])
            if isinstance(question, MultipleChoiceQA)
            else y_examples.iloc[i]
        )
        logging.debug(f"shot {i}: label={label}\tindex={y_examples.index[i]}")
        examples.append((X_examples.iloc[i], label))

    config = _build_config(
        task=task,
        question=question,
        add_task_description=True,
        custom_prompt_prefix=custom_prompt_prefix,
        prompt_variation=pv,
    )
    prompt = PromptBuilder(task).build_few_shot(
        row=row,
        config=config,
        examples=examples,
        few_shot_task_description=_get_few_shot_task_description(task),
        example_order=example_order,
    )
    logging.debug(prompt)
    return prompt


def encode_row_prompt_chat(
    row: pd.Series,
    task: TaskMetadata,
    tokenizer: AutoTokenizer,
    system_prompt: str | None = _DEFAULT,  # type: ignore[assignment]
    chat_prompt: str | None = _DEFAULT,  # type: ignore[assignment]
    numeric: bool = False,
    question: QAInterface | None = None,
    custom_prompt_prefix: str | None = None,
    custom_prompt_suffix: str | None = None,
    prompt_variation: dict | None = None,
) -> str:
    """Encode a row prompt using the tokenizer's chat template.

    Parameters
    ----------
    row : pd.Series
        The row that the question will be about.
    task : TaskMetadata
        The task metadata object.
    tokenizer : AutoTokenizer
        The tokenizer whose chat template will be applied.
    system_prompt : str | None, optional
        System prompt text. If omitted, the mode-appropriate default selected
        by `numeric` is used. Pass `None` explicitly to disable the system
        role (e.g. for Gemma-style templates that reject it).
    chat_prompt : str | None, optional
        Assistant prefill text. If omitted, the mode-appropriate default
        selected by `numeric` is used. Pass `None` explicitly to skip the
        assistant prefill — note that this routes inference through
        `add_generation_prompt=True` and breaks the last-token scoring
        assumption used by `LLMClassifier`, so it is not appropriate for the
        benchmark path.
    numeric : bool, optional
        Whether numeric risk prompting is being used. Selects which default
        prompts are applied when `system_prompt` / `chat_prompt` are omitted.
    question : QAInterface, optional
        The question interface to use.
    custom_prompt_prefix : str, optional
        A custom prefix to prepend before the task description.
    custom_prompt_suffix : str, optional
        A custom suffix to append after the row encoding (before the question).
    prompt_variation : dict | None, optional
        Prompt style overrides (format, connector, granularity, order, etc.).

    Returns
    -------
    str
        The fully formatted chat-template prompt.
    """
    if system_prompt is _DEFAULT:
        system_prompt = NUMERIC_SYSTEM_PROMPT if numeric else SYSTEM_PROMPT
    if chat_prompt is _DEFAULT:
        chat_prompt = NUMERIC_CHAT_PROMPT if numeric else ANTHROPIC_CHAT_PROMPT

    # Suppress answer prefix in the user message; it's supplied as the
    # assistant prefill turn so it must not appear twice.
    config = _build_config(
        task=task,
        question=question,
        custom_prompt_prefix=custom_prompt_prefix,
        custom_prompt_suffix=custom_prompt_suffix,
        prompt_variation=prompt_variation,
        system_prompt=system_prompt,
    )
    return PromptBuilder(task).build_chat(row[task.features], config, tokenizer, chat_prompt=chat_prompt)


def apply_chat_template(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str | None = None,
    chat_prompt: str | None = None,
    **kwargs,
) -> str:
    """Apply the tokenizer's chat template to assemble a single prompt string.

    Notes
    -----
    `system_prompt` is treated as "include" iff it is not `None`. This means an
    empty string `""` will inject an empty system message rather than be
    treated as "no system role" — pass `None` (or omit the argument) to skip
    the system role entirely.

    `chat_prompt` is the assistant prefill. When provided, the returned prompt
    is trimmed so it ends exactly with `chat_prompt`, preserving the
    last-token scoring contract relied on by `LLMClassifier`. If the chat
    template mutates or strips the prefill (so it cannot be located verbatim
    in the rendered output), a `ValueError` is raised rather than silently
    returning a corrupted prompt.

    When `chat_prompt is None`, `add_generation_prompt=True` is used and the
    model is left to generate freely; this is **not** appropriate for the
    benchmark scoring path (the last token will be a template-emitted role
    header, not the prefill).
    """
    # Add system prompt
    conversation = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []

    # Add user prompt
    conversation.append({"role": "user", "content": user_prompt})

    if chat_prompt is not None:
        # Using the Anthropic-style chat prompt
        conversation.append({"role": "assistant", "content": chat_prompt})
        kwargs.setdefault("add_generation_prompt", False)
    else:
        # No assistant prefill; let the model generate freely
        kwargs.setdefault("add_generation_prompt", True)

    # Apply prompt template
    filled_prompt = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        **kwargs,
    )

    if chat_prompt is not None:
        # Trim any special tokens that the template appended after the prefill
        # (e.g. a trailing newline or `<end_of_turn>`) so the last token of the
        # returned prompt is the last token of `chat_prompt` itself — this is
        # what `LLMClassifier` assumes when it reads answer-token probabilities.
        idx = filled_prompt.rfind(chat_prompt)
        if idx == -1:
            raise ValueError(
                "Assistant prefill not found verbatim in the templated output; "
                "the tokenizer's chat template likely transforms it (e.g. "
                "stripping or escaping). Cannot safely trim trailing tokens — "
                "pass a `chat_prompt` that survives templating, or run without "
                "an assistant prefill."
            )
        filled_prompt = filled_prompt[: idx + len(chat_prompt)]

    return filled_prompt
