"""Tests for the prompt variation framework in folktexts/prompting.py."""
from __future__ import annotations

import pytest

from folktexts.prompting import (
    DEFAULT_PROMPT_STYLE,
    VaryConnector,
    VaryFeatureOrder,
    VaryFormat,
    VaryPrefix,
    VarySuffix,
    VaryValueMap,
    encode_row_prompt,
    encode_row_prompt_few_shot,
    reset_building_block_cache,
)


class TestVaryFormat:

    @pytest.mark.parametrize("fmt,expected_start,expected_end", [
        ("bullet",     "- ",     "\n"),
        ("comma",      None,     ", "),
        ("text",       "The ",   ". "),
        ("textbullet", "- The ", ".\n"),
    ])
    def test_format(self, acs_income_task, acs_row, fmt, expected_start, expected_end):
        # VaryFormat must be applied after VaryValueMap -> VaryConnector (per source warning)
        row = VaryValueMap(task=acs_income_task, granularity="original")(acs_row.copy())
        row = VaryConnector(task=acs_income_task, connector="is")(row)
        vf = VaryFormat(task=acs_income_task, format=fmt)
        result = vf(row)
        print(result)
        for col in acs_income_task.features:
            val = result[col]
            if expected_start:
                assert val.startswith(expected_start), (
                    f"format={fmt!r}: expected start {expected_start!r}, got {val!r}"
                )
            assert val.endswith(expected_end), (
                f"format={fmt!r}: expected end {expected_end!r}, got {val!r}"
            )


class TestVaryConnector:

    @pytest.mark.parametrize("connector,expected_sep", [
        ("is",  " is "),
        ("=",   " = "),
        (":",   ": "),
    ])
    def test_connector(self, acs_income_task, acs_row, connector, expected_sep):
        # VaryConnector must be applied after VaryValueMap (per source warning)
        row = VaryValueMap(task=acs_income_task, granularity="original")(acs_row.copy())
        vc = VaryConnector(task=acs_income_task, connector=connector)
        result = vc(row)
        print(result)
        for col in acs_income_task.features:
            assert expected_sep in result[col], (
                f"connector={connector!r}: separator {expected_sep!r} not found in {result[col]!r}"
            )


class TestVaryValueMap:

    def test_original_returns_strings(self, acs_income_task, acs_row):
        result = VaryValueMap(task=acs_income_task, granularity="original")(acs_row.copy())
        print(result)
        for col in acs_income_task.features:
            assert isinstance(result[col], str), (
                f"Expected str for col={col!r}, got {type(result[col])}"
            )

    def test_original_age_exact(self, acs_income_task, acs_row):
        result = VaryValueMap(task=acs_income_task, granularity="original")(acs_row.copy())
        assert "years old" in result["AGEP"]
        # exact age: no dash (not a range)
        assert "-" not in result["AGEP"].split("years")[0]

    def test_original_wkhp_exact(self, acs_income_task, acs_row):
        result = VaryValueMap(task=acs_income_task, granularity="original")(acs_row.copy())
        assert "hours" in result["WKHP"]
        assert "-" not in result["WKHP"].split("hours")[0]

    def test_low_returns_strings(self, acs_income_task, acs_row):
        result = VaryValueMap(task=acs_income_task, granularity="low")(acs_row.copy())
        print(result)
        for col in acs_income_task.features:
            assert isinstance(result[col], str), (
                f"Expected str for col={col!r} with low granularity, got {type(result[col])}"
            )

    def test_low_age_is_range(self, acs_income_task, acs_row):
        result = VaryValueMap(task=acs_income_task, granularity="low")(acs_row.copy())
        # e.g. "30-39 years old" or "Less than 18 years old" or "90 or more years old"
        assert "years old" in result["AGEP"]
        age_part = result["AGEP"].split("years")[0].strip()
        is_range = "-" in age_part
        is_edge = age_part.startswith("Less than") or age_part.endswith("or more")
        assert is_range or is_edge, f"Expected age range, got {result['AGEP']!r}"

    def test_low_wkhp_is_range(self, acs_income_task, acs_row):
        result = VaryValueMap(task=acs_income_task, granularity="low")(acs_row.copy())
        # e.g. "30-39 hours" or "more than 60 hours" or N/A
        wkhp = result["WKHP"]
        is_range = "-" in wkhp.split("hours")[0]
        is_edge = wkhp.startswith("more than") or wkhp.startswith("N/A")
        assert is_range or is_edge, f"Expected hours range, got {wkhp!r}"


class TestVaryFeatureOrder:

    def test_reversed(self, acs_income_task, acs_row):
        features = acs_income_task.features
        reversed_order = list(reversed(features))
        result = VaryFeatureOrder(task=acs_income_task, order=reversed_order)(acs_row.copy())
        feature_set = set(features)
        result_features = [c for c in result.index if c in feature_set]
        assert result_features == reversed_order

    def test_none_leaves_row_unchanged(self, acs_income_task, acs_row):
        result = VaryFeatureOrder(task=acs_income_task, order=None)(acs_row.copy())
        assert list(result.index) == list(acs_row.index)


class TestVaryPrefix:

    def test_adds_prefix_key(self, acs_income_task, acs_row):
        vp = VaryPrefix(task=acs_income_task, add_task_description=True, task_description="Desc.\n")
        result = vp(acs_row.copy())
        assert "_PREFIX" in result.index

    def test_contains_task_description(self, acs_income_task, acs_row):
        desc = "Custom task description.\n"
        vp = VaryPrefix(task=acs_income_task, add_task_description=True, task_description=desc)
        result = vp(acs_row.copy())
        assert desc in result["_PREFIX"]


class TestVarySuffix:

    def test_adds_suffix_key(self, acs_income_task, acs_row):
        result = VarySuffix(task=acs_income_task)(acs_row.copy())
        assert "_SUFFIX" in result.index

    def test_contains_question_text(self, acs_income_task, acs_row):
        result = VarySuffix(task=acs_income_task)(acs_row.copy())
        assert acs_income_task.question.get_question_prompt() in result["_SUFFIX"]


class TestEncodeRowPrompt:

    def setup_method(self):
        reset_building_block_cache()

    def test_returns_nonempty_string(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(acs_row, task=acs_income_task)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_contains_question(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(acs_row, task=acs_income_task)
        assert acs_income_task.question.get_question_prompt() in prompt

    def test_contains_task_description(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(acs_row, task=acs_income_task, add_task_description=True)
        assert "survey" in prompt.lower()

    def test_different_formats_produce_different_prompts(self, acs_income_task, acs_row):
        prompt_bullet = encode_row_prompt(
            acs_row, task=acs_income_task,
            prompt_variation={**DEFAULT_PROMPT_STYLE, "format": "bullet"},
        )
        prompt_comma = encode_row_prompt(
            acs_row, task=acs_income_task,
            prompt_variation={**DEFAULT_PROMPT_STYLE, "format": "comma"},
        )
        assert prompt_bullet != prompt_comma

    def test_reset_cache_still_produces_valid_prompt(self, acs_income_task, acs_row):
        encode_row_prompt(acs_row, task=acs_income_task)
        reset_building_block_cache()
        prompt = encode_row_prompt(acs_row, task=acs_income_task)
        print(prompt)
        assert isinstance(prompt, str) and len(prompt) > 0


class TestEncodeRowPromptFewShot:

    def setup_method(self):
        reset_building_block_cache()

    def test_returns_string(self, acs_income_task, acs_income_dataset, acs_row):
        prompt = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            n_shots=1,
            reuse_examples=True,
        )
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_balanced_examples_contain_both_labels(
        self, acs_income_task, acs_income_dataset, acs_row
    ):
        """With compose_few_shot_examples='balanced', each label should appear equally."""
        n_shots = 2
        prompt = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            n_shots=n_shots,
            reuse_examples=True,
            compose_few_shot_examples="balanced",
        )
        print(prompt)
        answer_prefix = acs_income_task.question.get_answer_prefix()
        # Extract the answer key after each "Answer:" in the few-shot examples
        # (skip the final unanswered one at the end)
        import re
        answers = re.findall(rf"{re.escape(answer_prefix)}\s*(\w+)", prompt)
        print("Extracted few-shot answers:", answers)
        assert len(answers) == n_shots, f"Expected {n_shots} answers, got {answers}"
        unique_answers = set(answers)
        assert len(unique_answers) == 2, (
            f"Expected both labels in balanced examples, got {unique_answers}"
        )

    def test_question_appears_once_at_end(self, acs_income_task, acs_income_dataset, acs_row):
        """The question prompt should only appear at the end for the target row.
        Few-shot examples use only the answer key as suffix, not the full question."""
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