"""Tests for the prompt variation framework in folktexts/prompting.py."""

from __future__ import annotations

import re

import pytest
from folktexts.prompting import (
    DEFAULT_PROMPT_STYLE,
    FeatureItem,
    PromptBuilder,
    PromptConfig,
    VaryConnector,
    VaryFeatureOrder,
    VaryFormat,
    VaryOrder,
    VaryPrefix,
    VarySuffix,
    VarySystemPrompt,
    VaryValueMap,
    encode_row_prompt,
    encode_row_prompt_few_shot,
)


def _make_items(task, row) -> list[FeatureItem]:
    """Create FeatureItems from a task and row (pre-VaryValueMap)."""
    return [
        FeatureItem(col=col, label=task.cols_to_text[col].short_description, raw_value=row[col])
        for col in task.features
        if col in row.index
    ]


class TestVaryValueMap:
    def test_original_returns_strings(self, acs_income_task, acs_row):
        items = _make_items(acs_income_task, acs_row)
        result = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        for item in result:
            assert isinstance(item.text_value, str), f"Expected str for col={item.col!r}, got {type(item.text_value)}"

    def test_original_age_exact(self, acs_income_task, acs_row):
        items = _make_items(acs_income_task, acs_row)
        result = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        agep = next(i for i in result if i.col == "AGEP")
        assert "years old" in agep.text_value
        assert "-" not in agep.text_value.split("years")[0]

    def test_original_wkhp_exact(self, acs_income_task, acs_row):
        items = _make_items(acs_income_task, acs_row)
        result = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        wkhp = next(i for i in result if i.col == "WKHP")
        assert "hours" in wkhp.text_value
        assert "-" not in wkhp.text_value.split("hours")[0]

    def test_low_returns_strings(self, acs_income_task, acs_row):
        from folktexts.acs.acs_columns_alt import simplified_value_maps

        items = _make_items(acs_income_task, acs_row)
        vm = VaryValueMap.with_low_granularity(acs_income_task.cols_to_text, simplified_value_maps)
        result = vm(items)
        for item in result:
            assert isinstance(item.text_value, str), (
                f"Expected str for col={item.col!r} with low granularity, got {type(item.text_value)}"
            )

    def test_low_age_is_range(self, acs_income_task, acs_row):
        from folktexts.acs.acs_columns_alt import simplified_value_maps

        items = _make_items(acs_income_task, acs_row)
        vm = VaryValueMap.with_low_granularity(acs_income_task.cols_to_text, simplified_value_maps)
        result = vm(items)
        agep = next(i for i in result if i.col == "AGEP")
        assert "years old" in agep.text_value
        age_part = agep.text_value.split("years")[0].strip()
        is_range = "-" in age_part
        is_edge = age_part.startswith("Less than") or age_part.endswith("or more")
        assert is_range or is_edge, f"Expected age range, got {agep.text_value!r}"

    def test_low_wkhp_is_range(self, acs_income_task, acs_row):
        from folktexts.acs.acs_columns_alt import simplified_value_maps

        items = _make_items(acs_income_task, acs_row)
        vm = VaryValueMap.with_low_granularity(acs_income_task.cols_to_text, simplified_value_maps)
        result = vm(items)
        wkhp = next(i for i in result if i.col == "WKHP")
        is_range = "-" in wkhp.text_value.split("hours")[0]
        is_edge = wkhp.text_value.startswith("more than") or wkhp.text_value.startswith("N/A")
        assert is_range or is_edge, f"Expected hours range, got {wkhp.text_value!r}"

    def test_with_low_granularity_does_not_mutate_task(self, acs_income_task, acs_row):
        from folktexts.acs.acs_columns_alt import simplified_value_maps

        original_map = acs_income_task.cols_to_text["AGEP"]._value_map
        VaryValueMap.with_low_granularity(acs_income_task.cols_to_text, simplified_value_maps)
        assert acs_income_task.cols_to_text["AGEP"]._value_map is original_map


class TestVaryOrder:
    def test_reversed(self, acs_income_task, acs_row):
        features = acs_income_task.features
        reversed_order = list(reversed(features))
        items = _make_items(acs_income_task, acs_row)
        result = VaryOrder(order=reversed_order)(items)
        assert [i.col for i in result] == reversed_order

    def test_none_leaves_order_unchanged(self, acs_income_task, acs_row):
        items = _make_items(acs_income_task, acs_row)
        result = VaryOrder(order=None)(items)
        assert [i.col for i in result] == [i.col for i in items]

    def test_alias(self):
        assert VaryFeatureOrder is VaryOrder


class TestVaryConnector:
    @pytest.mark.parametrize(
        "connector,expected_sep",
        [
            ("is", " is "),
            ("=", " = "),
            (":", ": "),
        ],
    )
    def test_connector(self, acs_income_task, acs_row, connector, expected_sep):
        items = _make_items(acs_income_task, acs_row)
        items = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        result = VaryConnector(connector=connector)(items)
        for item in result:
            assert expected_sep in item.connected, (
                f"connector={connector!r}: separator {expected_sep!r} not found in {item.connected!r}"
            )


class TestVaryFormat:
    @pytest.mark.parametrize(
        "fmt,expected_start,expected_end",
        [
            ("bullet", "- ", "\n"),
            ("comma", None, ", "),
            ("text", "The ", ". "),
            ("textbullet", "- The ", ".\n"),
        ],
    )
    def test_format(self, acs_income_task, acs_row, fmt, expected_start, expected_end):
        items = _make_items(acs_income_task, acs_row)
        items = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        items = VaryConnector(connector="is")(items)
        result = VaryFormat(format=fmt)(items)
        assert isinstance(result, str)
        if expected_start:
            assert result.startswith(expected_start), (
                f"format={fmt!r}: expected start {expected_start!r}, got beginning {result[:20]!r}"
            )
        assert result.endswith(expected_end), f"format={fmt!r}: expected end {expected_end!r}, got end {result[-20:]!r}"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            VaryFormat(format="invalid_format")


class TestVarySystemPrompt:
    def test_returns_system_prompt_string(self):
        sp = "You are a helpful assistant."
        vsp = VarySystemPrompt(system_prompt=sp)
        assert vsp() == sp

    def test_in_prompt_config(self, acs_income_task):
        sp = "System instruction."
        config = PromptConfig.default(acs_income_task)
        assert config.system_prompt is None
        config_with_sp = PromptConfig(
            prefix=config.prefix,
            value_map=config.value_map,
            order=config.order,
            connector=config.connector,
            format=config.format,
            suffix=config.suffix,
            system_prompt=VarySystemPrompt(system_prompt=sp),
        )
        assert config_with_sp.system_prompt() == sp


class TestVaryPrefix:
    def test_contains_task_description(self, acs_income_task):
        desc = "Custom task description.\n"
        vp = VaryPrefix(task_description=desc, add_task_description=True)
        result = vp()
        assert desc in result

    def test_no_task_description(self, acs_income_task):
        vp = VaryPrefix(task_description="Some desc.\n", add_task_description=False)
        result = vp()
        assert "Information:" in result
        assert "Some desc." not in result

    def test_custom_prefix_appended(self, acs_income_task):
        vp = VaryPrefix(task_description="Desc.\n", add_task_description=True, custom_prefix="Extra context.")
        result = vp()
        assert "Extra context." in result
        assert "Desc." in result


class TestVarySuffix:
    def test_contains_question_text(self, acs_income_task):
        vs = VarySuffix(question=acs_income_task.question, show_question=True)
        result = vs()
        assert acs_income_task.question.get_question_prompt() in result

    def test_show_question_false_uses_answer_prefix(self, acs_income_task):
        vs = VarySuffix(question=acs_income_task.question, show_question=False)
        result = vs()
        assert acs_income_task.question.get_answer_prefix() in result
        assert acs_income_task.question.get_question_prompt() not in result

    def test_show_label(self, acs_income_task):
        vs = VarySuffix(question=acs_income_task.question, show_question=False, show_label=True, label="A")
        result = vs()
        assert " A" in result


class TestPromptBuilder:
    def test_build_returns_nonempty_string(self, acs_income_task, acs_row):
        config = PromptConfig.default(acs_income_task)
        prompt = PromptBuilder(acs_income_task).build(acs_row, config)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_build_contains_question(self, acs_income_task, acs_row):
        config = PromptConfig.default(acs_income_task)
        prompt = PromptBuilder(acs_income_task).build(acs_row, config)
        assert acs_income_task.question.get_question_prompt() in prompt

    def test_build_contains_task_description(self, acs_income_task, acs_row):
        config = PromptConfig.default(acs_income_task)
        prompt = PromptBuilder(acs_income_task).build(acs_row, config)
        assert "survey" in prompt.lower()


class TestEncodeRowPrompt:
    def test_returns_nonempty_string(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(acs_row, task=acs_income_task)
        print(f"\n--- default ---\n{prompt}")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_contains_question(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(acs_row, task=acs_income_task)
        assert acs_income_task.question.get_question_prompt() in prompt

    def test_contains_task_description(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(acs_row, task=acs_income_task, add_task_description=True)
        assert "survey" in prompt.lower()

    def test_different_formats_produce_different_prompts(self, acs_income_task, acs_row):
        prompt_bullet = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_variation={**DEFAULT_PROMPT_STYLE, "format": "bullet"},
        )
        prompt_comma = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_variation={**DEFAULT_PROMPT_STYLE, "format": "comma"},
        )
        print(f"\n--- bullet ---\n{prompt_bullet}")
        print(f"\n--- comma ---\n{prompt_comma}")
        assert prompt_bullet != prompt_comma

    @pytest.mark.parametrize("connector", ["is", "=", ":"])
    def test_connector_variation(self, acs_income_task, acs_row, connector):
        prompt = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_variation={**DEFAULT_PROMPT_STYLE, "connector": connector},
        )
        print(f"\n--- connector={connector!r} ---\n{prompt}")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_low_granularity_variation(self, acs_income_task, acs_row):
        prompt_orig = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_variation={**DEFAULT_PROMPT_STYLE, "granularity": "original"},
        )
        prompt_low = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_variation={**DEFAULT_PROMPT_STYLE, "granularity": "low"},
        )
        print(f"\n--- granularity=original ---\n{prompt_orig}")
        print(f"\n--- granularity=low ---\n{prompt_low}")
        assert isinstance(prompt_low, str) and len(prompt_low) > 0
        assert prompt_orig != prompt_low

    def test_order_variation(self, acs_income_task, acs_row):
        features = acs_income_task.features
        reversed_order = list(reversed(features))
        prompt_default = encode_row_prompt(acs_row, task=acs_income_task)
        prompt_reversed = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_variation={**DEFAULT_PROMPT_STYLE, "order": reversed_order},
        )
        print(f"\n--- order=default ---\n{prompt_default}")
        print(f"\n--- order=reversed ---\n{prompt_reversed}")
        assert prompt_default != prompt_reversed

    def test_custom_prompt_prefix(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            custom_prompt_prefix="Extra context here.",
        )
        print(f"\n--- custom_prompt_prefix ---\n{prompt}")
        assert "Extra context here." in prompt

    def test_custom_prompt_suffix(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            custom_prompt_suffix=" [end]",
        )
        print(f"\n--- custom_prompt_suffix ---\n{prompt}")
        assert prompt.endswith(" [end]")


class TestEncodeRowPromptFewShot:
    @pytest.mark.parametrize("composition", ["random", "balanced"])
    def test_returns_string(self, acs_income_task, acs_income_dataset, acs_row, composition):
        prompt = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            n_shots=2,
            reuse_examples=True,
            compose_few_shot_examples=composition,
        )
        print(f"\n--- few-shot (2 shots, composition={composition!r}) ---\n{prompt}")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_balanced_examples_contain_both_labels(self, acs_income_task, acs_income_dataset, acs_row):
        n_shots = 2
        prompt = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            n_shots=n_shots,
            reuse_examples=True,
            compose_few_shot_examples="balanced",
        )
        print(f"\n--- few-shot balanced ---\n{prompt}")
        answer_prefix = acs_income_task.question.get_answer_prefix()
        answers = re.findall(rf"{re.escape(answer_prefix)}\s*(\w+)", prompt)
        assert len(answers) == n_shots, f"Expected {n_shots} answers, got {answers}"
        assert len(set(answers)) == 2, f"Expected both labels in balanced examples, got {set(answers)}"

    def test_question_appears_once_at_end(self, acs_income_task, acs_income_dataset, acs_row):
        prompt = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            n_shots=2,
            reuse_examples=False,
        )
        question_text = acs_income_task.question.get_question_prompt()
        assert prompt.count(question_text) == 1
        assert prompt.endswith(question_text)
