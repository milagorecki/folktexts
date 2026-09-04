"""Integration tests for the Benchmark and TransformersLLMClassifier pipeline.

Uses the tiny-random-gpt2 model and the 10-row ACS fixture for fast runs.
The tiny model produces meaningless logits; tests assert shape/range correctness
and structural properties (e.g. that order-bias correction generates distinct
prompts per permutation).
"""

from __future__ import annotations

import json
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
from folktexts.benchmark import Benchmark, BenchmarkConfig
from folktexts.classifier import TransformersLLMClassifier
from folktexts.col_to_text import ColumnToText
from folktexts.prompting import FewShotConfig, PromptConfig, encode_row_prompt_few_shot
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA, TextNumericQA
from folktexts.task import TaskMetadata


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer(causal_lm_name_or_path):
    from folktexts.llm_utils import load_model_tokenizer

    return load_model_tokenizer(causal_lm_name_or_path)


@pytest.fixture(scope="module")
def clf(tiny_model_and_tokenizer, acs_income_task):
    model, tokenizer = tiny_model_and_tokenizer
    return TransformersLLMClassifier(
        model=model,
        tokenizer=tokenizer,
        task=acs_income_task,
        correct_order_bias=True,
        batch_size=3,
        context_size=256,
    )


@pytest.fixture(scope="module")
def clf_no_bias(tiny_model_and_tokenizer, acs_income_task):
    model, tokenizer = tiny_model_and_tokenizer
    return TransformersLLMClassifier(
        model=model,
        tokenizer=tokenizer,
        task=acs_income_task,
        correct_order_bias=False,
        batch_size=3,
        context_size=256,
    )


class TestPredictProba:
    def test_output_shape(self, clf, acs_income_dataset):
        X_test, _ = acs_income_dataset.get_test()
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_output_range(self, clf, acs_income_dataset):
        X_test, _ = acs_income_dataset.get_test()
        proba = clf.predict_proba(X_test)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_rows_sum_to_one(self, clf, acs_income_dataset):
        X_test, _ = acs_income_dataset.get_test()
        proba = clf.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_few_shot(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset):
        model, tokenizer = tiny_model_and_tokenizer
        X_test, _ = acs_income_dataset.get_test()
        prompt_config = PromptConfig.from_dict({}, task=acs_income_task)
        encode_row_fn = partial(
            encode_row_prompt_few_shot,
            task=acs_income_task,
            dataset=acs_income_dataset,
            prompt_config=prompt_config,
            few_shot_config=FewShotConfig(n_shots=2, reuse_examples=True),
        )
        clf_fs = TransformersLLMClassifier(
            model=model,
            tokenizer=tokenizer,
            task=acs_income_task,
            encode_row=encode_row_fn,
            correct_order_bias=True,
            batch_size=3,
            context_size=512,
        )
        proba = clf_fs.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predictions_saved_to_csv(self, clf, acs_income_dataset, tmp_path):
        """predict_proba with predictions_save_path writes a CSV with scores and labels."""
        import pandas as pd

        X_test, y_test = acs_income_dataset.get_test()
        save_path = tmp_path / "preds.csv"
        clf.predict_proba(X_test, predictions_save_path=save_path, labels=y_test)
        assert save_path.exists()
        df = pd.read_csv(save_path, index_col=0)
        assert "risk_score" in df.columns
        assert "label" in df.columns
        assert len(df) == len(X_test)

    def test_predictions_loaded_from_disk(self, clf, acs_income_dataset, tmp_path):
        """predict_proba returns cached scores without re-running inference when the file exists."""
        X_test, y_test = acs_income_dataset.get_test()
        save_path = tmp_path / "preds_cached.csv"
        proba1 = clf.predict_proba(X_test, predictions_save_path=save_path, labels=y_test)

        with patch.object(
            clf,
            "_query_prompt_risk_estimates_batch",
            side_effect=AssertionError("inference should not run when cache exists"),
        ):
            proba2 = clf.predict_proba(X_test, predictions_save_path=save_path, labels=y_test)

        np.testing.assert_array_equal(proba1, proba2)


class TestClassifierOrderBias:
    def test_distinct_prompts_per_permutation(self, clf, acs_income_dataset):
        """Each MCQ permutation must produce a distinct prompt — guards against
        the bug where `question` was silently dropped when `prompt_config` was set."""
        X_test, _ = acs_income_dataset.get_test()

        captured: list[tuple] = []  # (question, prompt)
        original_encode_row = clf.encode_row

        def capturing_encode_row(row, **kwargs):
            prompt = original_encode_row(row, **kwargs)
            captured.append((kwargs.get("question"), prompt))
            return prompt

        clf._encode_row = capturing_encode_row
        try:
            clf.predict_proba(X_test)
        finally:
            clf._encode_row = original_encode_row

        # With 2-choice MCQ there are 2 permutations → 2 captures per row
        n_rows = len(X_test)
        assert len(captured) == n_rows * 2, (
            f"Expected {n_rows * 2} encode_row calls (2 permutations × {n_rows} rows), got {len(captured)}"
        )

        # For each row, the two prompts (one per permutation) must differ.
        # Loop order is batch-outer, question-inner, rows-innermost:
        # captured[0..n_rows-1] = all rows under permutation 0,
        # captured[n_rows..2*n_rows-1] = all rows under permutation 1.
        for i in range(n_rows):
            q0, p0 = captured[i]
            q1, p1 = captured[n_rows + i]
            assert q0 != q1, f"Row {i}: question objects are identical across permutations"
            assert p0 != p1, f"Row {i}: prompts are identical across permutations — question override was dropped"

    def test_no_bias_correction_single_prompt_per_row(self, clf_no_bias, acs_income_dataset):
        X_test, _ = acs_income_dataset.get_test()

        captured: list[str] = []
        original_encode_row = clf_no_bias.encode_row

        def capturing_encode_row(row, **kwargs):
            prompt = original_encode_row(row, **kwargs)
            captured.append(prompt)
            return prompt

        clf_no_bias._encode_row = capturing_encode_row
        try:
            clf_no_bias.predict_proba(X_test)
        finally:
            clf_no_bias._encode_row = original_encode_row

        assert len(captured) == len(X_test)

    def test_batch_path_prompts_match_series_path(self, clf, acs_income_dataset):
        """The batch loop materializes rows once per batch instead of re-iterating
        `batch_data.iterrows()` inside the per-question loop. This test guards the
        byte-equality of the emitted prompts against a Series-per-row reference."""
        X_test, _ = acs_income_dataset.get_test()

        captured: list[tuple] = []  # (question, prompt)
        original_encode_row = clf.encode_row

        def capturing_encode_row(row, **kwargs):
            prompt = original_encode_row(row, **kwargs)
            captured.append((kwargs.get("question"), prompt))
            return prompt

        clf._encode_row = capturing_encode_row
        try:
            clf.predict_proba(X_test)
        finally:
            clf._encode_row = original_encode_row

        # Two permutations under order-bias correction.
        n_rows = len(X_test)
        assert len(captured) == n_rows * 2

        # Reference: encode each row directly with the classifier's encode_row,
        # feeding pd.Series produced by df.iloc[i]. This mimics the pre-refactor
        # path (Series per row) and must byte-match the batch path.
        for perm_idx in range(2):
            slice_start = perm_idx * n_rows
            perm_captures = captured[slice_start : slice_start + n_rows]
            # All prompts in this slice share the same question object.
            question = perm_captures[0][0]
            for i in range(n_rows):
                q_captured, prompt_captured = perm_captures[i]
                assert q_captured is question
                prompt_ref = original_encode_row(X_test.iloc[i], question=question)
                assert prompt_captured == prompt_ref, f"Prompt mismatch at row {i}, permutation {perm_idx}"


class TestClassifierConstruction:
    def test_rejects_removed_kwargs(self, tiny_model_and_tokenizer, acs_income_task):
        """Removed prompt-shaping kwargs (e.g. custom_prompt_prefix) used to be silently
        swallowed into inference_kwargs; now they raise with a pointer to prompt_config."""
        model, tokenizer = tiny_model_and_tokenizer
        with pytest.raises(TypeError, match="custom_prompt_prefix"):
            TransformersLLMClassifier(
                model=model,
                tokenizer=tokenizer,
                task=acs_income_task,
                custom_prompt_prefix="extra context",
            )


class TestBenchmarkConfig:
    def test_default_is_hashable(self):
        cfg = BenchmarkConfig()
        assert isinstance(hash(cfg), int)

    def test_distinct_configs_have_different_hashes(self):
        assert hash(BenchmarkConfig()) != hash(BenchmarkConfig(seed=99))

    def test_hash_with_few_shot_config(self):
        cfg = BenchmarkConfig(few_shot_config=FewShotConfig(n_shots=2))
        assert isinstance(hash(cfg), int)

    def test_few_shot_hash_is_deterministic_across_processes(self):
        """B4 regression: __hash__ hashed few_shot_config with Python's salted builtin
        hash(), so `results.bench-{hash}.json` got a different name every process. The
        few-shot hash must be stable across PYTHONHASHSEED values."""
        import os
        import subprocess
        import sys

        code = (
            "from folktexts.benchmark import BenchmarkConfig;"
            "from folktexts.prompting import FewShotConfig;"
            "print(hash(BenchmarkConfig(few_shot_config=FewShotConfig(n_shots=2))))"
        )

        def _hash_with_seed(seed: int) -> str:
            return subprocess.check_output(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
            ).strip()

        hashes = {_hash_with_seed(seed) for seed in (1, 2)}
        assert len(hashes) == 1, f"few-shot config hash is not deterministic: {hashes}"

    def test_hash_with_feature_subset(self):
        cfg = BenchmarkConfig(feature_subset=["AGEP", "WKHP"])
        assert isinstance(hash(cfg), int)

    def test_hash_with_prompt_variation(self):
        cfg = BenchmarkConfig(prompt_variation={"format": "bullet"})
        assert isinstance(hash(cfg), int)

    def test_hash_differs_with_few_shot_config(self):
        cfg_zero_shot = BenchmarkConfig()
        cfg_few_shot = BenchmarkConfig(few_shot_config=FewShotConfig(n_shots=2))
        assert hash(cfg_zero_shot) != hash(cfg_few_shot)

    def test_hash_differs_with_reasoning(self):
        assert hash(BenchmarkConfig()) != hash(BenchmarkConfig(reasoning="low"))
        assert hash(BenchmarkConfig(reasoning="low")) != hash(BenchmarkConfig(reasoning="high"))

    def test_save_load_roundtrip(self, tmp_path):
        cfg = BenchmarkConfig(seed=7, batch_size=4)
        path = tmp_path / "config.json"
        cfg.save_to_disk(path)
        loaded = BenchmarkConfig.load_from_disk(path)
        assert cfg == loaded

    def test_save_load_with_few_shot_config(self, tmp_path):
        """FewShotConfig nested object survives a save/load roundtrip."""
        few_shot = FewShotConfig(n_shots=3, compose="balanced", reuse_examples=True)
        cfg = BenchmarkConfig(few_shot_config=few_shot)
        path = tmp_path / "config_fs.json"
        cfg.save_to_disk(path)
        loaded = BenchmarkConfig.load_from_disk(path)
        assert loaded.few_shot_config == few_shot

    def test_save_load_null_few_shot_config(self, tmp_path):
        """None few_shot_config round-trips correctly (not reconstructed as FewShotConfig)."""
        cfg = BenchmarkConfig()
        path = tmp_path / "config_null_fs.json"
        cfg.save_to_disk(path)
        loaded = BenchmarkConfig.load_from_disk(path)
        assert loaded.few_shot_config is None

    def test_update_applies_known_keys(self):
        cfg = BenchmarkConfig()
        updated = cfg.update(seed=99, reasoning="low")
        assert updated.seed == 99
        assert updated.reasoning == "low"

    def test_load_legacy_few_shot_keys(self, tmp_path):
        """Back-compat: pre-refactor configs used flat few_shot/reuse/balance keys and
        had no few_shot_config; loading them used to raise TypeError. Stray result-file
        metadata (e.g. roc_auc) must also be tolerated."""
        legacy = {
            "numeric_risk_prompting": False,
            "few_shot": 3,
            "reuse_few_shot_examples": True,
            "balance_few_shot_examples": True,
            "seed": 7,
            "roc_auc": 0.81,  # stray metadata -> ignored, not a TypeError
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy))
        cfg = BenchmarkConfig.load_from_disk(path)
        assert cfg.few_shot_config == FewShotConfig(n_shots=3, reuse_examples=True, compose="balanced")
        assert cfg.seed == 7

    def test_update_ignores_unknown_keys(self):
        cfg = BenchmarkConfig()
        updated = cfg.update(nonexistent_key="value")
        assert updated == cfg

    def test_update_returns_new_object(self):
        cfg = BenchmarkConfig()
        updated = cfg.update(seed=1)
        assert updated is not cfg

    def test_reasoning_field_preserved(self):
        for value in ("low", "medium", "high", "0", "1024"):
            cfg = BenchmarkConfig(reasoning=value)
            assert cfg.reasoning == value

    def test_use_generated_text_field(self):
        cfg = BenchmarkConfig(use_generated_text=True)
        assert cfg.use_generated_text is True
        assert BenchmarkConfig().use_generated_text is False

    def test_prompt_variation_preserved(self):
        pv = {"format": "bullet", "connector": ":"}
        cfg = BenchmarkConfig(prompt_variation=pv)
        assert cfg.prompt_variation == pv

    def test_no_chat_prompt_warning_on_default_prompts(self, caplog):
        """Spin-off of B1: the chat-only warning used `is not None`, but the defaults
        are the PROMPT_DEFAULT sentinel (not None), so it fired on every default run."""
        import logging

        with caplog.at_level(logging.WARNING):
            Benchmark._validate_config(BenchmarkConfig(use_chat_template=False))
        assert "will be ignored" not in caplog.text

    def test_chat_prompt_warning_when_user_set_without_chat_template(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            Benchmark._validate_config(BenchmarkConfig(use_chat_template=False, system_prompt="custom"))
        assert "will be ignored" in caplog.text


class TestBenchmarkRun:
    """End-to-end tests for Benchmark.make_benchmark() + Benchmark.run().

    Each test constructs its own Benchmark to avoid shared mutable state
    (make_benchmark can mutate task fields like use_text_output_for_qa).
    """

    def _make_bench(self, model, tokenizer, task, dataset, **config_overrides):
        config_params = {"batch_size": 3, "context_size": 256}
        config_params.update(config_overrides)
        config = BenchmarkConfig(**config_params)
        return Benchmark.make_benchmark(
            task=task,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

    def test_run_produces_results(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None
        assert isinstance(bench.results, dict)

    def test_results_has_expected_keys(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        for key in ("accuracy", "roc_auc", "model_name", "threshold_fitted_on"):
            assert key in bench.results, f"Missing key '{key}' in results"

    def test_test_scores_shape_and_range(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        X_test, _ = acs_income_dataset.get_test()
        assert bench._y_test_scores is not None
        assert len(bench._y_test_scores) == len(X_test)
        assert np.all(bench._y_test_scores >= 0) and np.all(bench._y_test_scores <= 1)

    def test_save_results_writes_json(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        bench.save_results()
        result_files = list(bench.results_dir.glob("results.bench-*.json"))
        assert len(result_files) == 1
        with open(result_files[0]) as f:
            saved = json.load(f)
        assert "accuracy" in saved

    def test_save_results_content_matches_results_dict(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        bench.save_results()
        result_file = next(bench.results_dir.glob("results.bench-*.json"))
        with open(result_file) as f:
            saved = json.load(f)
        assert saved["accuracy"] == pytest.approx(bench.results["accuracy"])

    def test_run_with_few_shot_config(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            context_size=512,
            few_shot_config=FewShotConfig(n_shots=2, reuse_examples=True),
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None
        X_test, _ = acs_income_dataset.get_test()
        assert len(bench._y_test_scores) == len(X_test)

    def test_run_with_balanced_few_shot(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            context_size=512,
            few_shot_config=FewShotConfig(n_shots=2, compose="balanced", reuse_examples=True),
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None

    def test_run_with_per_class_few_shot(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        """B4 (related): per-class `compose` is normalized to a tuple by FewShotConfig,
        but Dataset.sample_n_train_examples used to accept only list/str -> the documented
        per-class few-shot feature crashed end-to-end. A tuple compose must run."""
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            context_size=512,
            few_shot_config=FewShotConfig(n_shots=2, compose=[1, 1], reuse_examples=True),
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None

    def test_run_with_prompt_variation(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            prompt_variation={"format": "bullet", "connector": "is"},
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None

    def test_run_no_order_bias(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            correct_order_bias=False,
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None

    def test_fit_threshold(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path, fit_threshold=3)
        assert bench.llm_clf._threshold_fitted_on == 3
        assert bench.results["threshold_fitted_on"] == 3

    def test_benchmark_hash_is_stable(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        assert hash(bench) == hash(bench)

    def test_benchmark_config_round_trips_via_hash(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset):
        """Two Benchmark objects built from identical configs share the same hash."""
        model, tokenizer = tiny_model_and_tokenizer
        bench1 = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset, seed=7)
        bench2 = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset, seed=7)
        assert hash(bench1) == hash(bench2)

    def test_different_configs_differ_in_hash(self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset):
        model, tokenizer = tiny_model_and_tokenizer
        bench1 = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset, seed=1)
        bench2 = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset, seed=2)
        assert hash(bench1) != hash(bench2)

    def test_benchmark_hash_stable_across_processes(self, causal_lm_name_or_path, acs_income_task, acs_income_dataset):
        """Hash must be identical across Python processes with different PYTHONHASHSEED.

        Tests within a single process always share the same seed, so they cannot
        catch hash randomization bugs.  This test spawns subprocesses with
        explicitly different seeds and compares their outputs.

        Covers the full hash chain: TransformersLLMClassifier → PromptConfig →
        VaryPrefix / VarySuffix / VaryValueMap / VaryOrder / VaryConnector /
        VaryFormat / VarySystemPrompt, which is where PYTHONHASHSEED bugs live.
        """
        import os
        import subprocess
        import sys
        import textwrap
        from pathlib import Path

        fixture_path = Path(__file__).parent / "acs_income_10rows.csv"
        if not fixture_path.exists():
            pytest.skip(f"ACS fixture not found at {fixture_path}. Run `python tests/create_acs_fixture.py` to generate it.")

        # Hash the full Benchmark (includes classifier → PromptConfig → Vary* chain)
        # using the tiny model and the fixture CSV so it's fast.
        bench_script = textwrap.dedent("""
        import pandas as pd
        from folktexts.acs import ACSTaskMetadata
        from folktexts.dataset import Dataset
        from folktexts.llm_utils import load_model_tokenizer
        from folktexts.classifier import TransformersLLMClassifier
        from folktexts.benchmark import Benchmark, BenchmarkConfig

        task = ACSTaskMetadata.get_task("ACSIncome", use_numeric_qa=False)
        df = pd.read_csv({fixture_path!r}, index_col=0)
        dataset = Dataset(data=df, task=task, test_size=0.3, val_size=0.0, seed=42)
        model, tokenizer = load_model_tokenizer({model!r})
        clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=task)
        bench = Benchmark(llm_clf=clf, dataset=dataset, config=BenchmarkConfig.default_config())
        print(hash(bench))
        """).format(fixture_path=str(fixture_path), model=causal_lm_name_or_path)

        def run(seed: int) -> str:
            env = {**os.environ, "PYTHONHASHSEED": str(seed)}
            result = subprocess.run(
                [sys.executable, "-c", bench_script],
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0, result.stderr
            return result.stdout.strip()

        h0, h1, h999 = run(0), run(1), run(999)
        assert h0 == h1 == h999, (
            f"Benchmark hash is not stable across processes: seed=0 → {h0}, seed=1 → {h1}, seed=999 → {h999}"
        )


class TestTemperatureWiring:
    """The transformers text extraction path must thread the resolved temperature + seed
    into `generate_text_batch` (greedy for non-reasoning models, 1.0 in thinking mode;
    explicit override wins)."""

    def _run_generated_text(self, tiny_model_and_tokenizer, acs_income_task, **clf_kwargs):
        """Run the transformers generated-text path with `generate_text_batch`
        patched to capture the kwargs it is called with (and return a canned
        response). Thinking is a classifier `reasoning=` kwarg here, not a QA field.
        """
        model, tokenizer = tiny_model_and_tokenizer
        captured: dict = {}

        def fake_generate(text_inputs, **kwargs):
            captured.update(kwargs)
            # generate_text_batch now returns the raw per-row generation
            # ({"text", "token_ids"}); the classifier splits it at its call site.
            return [{"text": "Probability: 40%", "token_ids": []} for _ in text_inputs]

        with patch(
            "folktexts.classifier.transformers_classifier.generate_text_batch",
            side_effect=fake_generate,
        ):
            clf = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=acs_income_task, **clf_kwargs)
            question = TextNumericQA(column="PINCP", text="dummy")
            risks, _ = clf._query_prompt_risk_estimates_batch(prompts_batch=["p"], question=question)
        return captured, risks

    def test_generated_text_defaults_to_greedy_and_threads_seed(self, tiny_model_and_tokenizer, acs_income_task):
        # Generation reads the decoupled `generation_seed`, not the init `seed`.
        captured, risks = self._run_generated_text(tiny_model_and_tokenizer, acs_income_task, generation_seed=7)
        assert captured["temperature"] == 0.0
        assert captured["seed"] == 7
        assert risks[0] == pytest.approx(0.4)

    def test_thinking_mode_defaults_to_temperature_one(self, tiny_model_and_tokenizer, acs_income_task):
        captured, _ = self._run_generated_text(tiny_model_and_tokenizer, acs_income_task, reasoning="high")
        assert captured["temperature"] == 1.0

    def test_explicit_temperature_override_reaches_generation(self, tiny_model_and_tokenizer, acs_income_task):
        captured, _ = self._run_generated_text(tiny_model_and_tokenizer, acs_income_task, temperature=0.7)
        assert captured["temperature"] == 0.7

    def test_temperature_changes_classifier_hash(self, tiny_model_and_tokenizer, acs_income_task):
        """An explicit temperature must produce distinct result-cache identity."""
        model, tokenizer = tiny_model_and_tokenizer
        clf_default = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=acs_income_task)
        clf_override = TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=acs_income_task, temperature=0.7)
        assert hash(clf_default) != hash(clf_override)

    def test_generation_seed_changes_hash_only_on_generated_text_path(self, tiny_model_and_tokenizer, acs_income_task):
        """`generation_seed` only affects outputs when text is sampled, so it must
        enter result identity on the generated-text path and be ignored on the
        deterministic token-probability path (no redundant result folders)."""
        model, tokenizer = tiny_model_and_tokenizer

        # Token-probability path: generation_seed is a no-op -> same hash.
        acs_income_task.use_text_output_for_qa = False
        h_logprob_a = hash(TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=acs_income_task, generation_seed=1))
        h_logprob_b = hash(TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=acs_income_task, generation_seed=2))
        assert h_logprob_a == h_logprob_b

        # Generated-text path: generation_seed changes sampling -> distinct hash.
        acs_income_task.use_text_output_for_qa = True
        h_text_a = hash(TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=acs_income_task, generation_seed=1))
        h_text_b = hash(TransformersLLMClassifier(model=model, tokenizer=tokenizer, task=acs_income_task, generation_seed=2))
        assert h_text_a != h_text_b


# ----------------------------------------------------------------------
# Benchmark._validate_config — chain-of-thought (CoT) prompting rules.
# ----------------------------------------------------------------------


class TestValidateConfigCoT:
    def test_rejects_cot_without_generated_text(self):
        # The CoT instruction is only appended on the generated-text path; on the
        # logprob path it is a silent no-op, so require the text path explicitly.
        with pytest.raises(ValueError, match="generated"):
            Benchmark._validate_config(BenchmarkConfig(cot_prompting=True, use_generated_text=False))

    def test_accepts_cot_with_generated_text(self):
        Benchmark._validate_config(BenchmarkConfig(cot_prompting=True, use_generated_text=True))

    def test_accepts_chat_template_with_cot(self):
        # Upstream rejected chat_template + CoT because CoT applied the chat
        # template internally (double-wrap). Here everything routes through
        # `prompting.apply_chat_template`, so the combination is valid.
        Benchmark._validate_config(BenchmarkConfig(use_chat_template=True, cot_prompting=True, use_generated_text=True))


# ----------------------------------------------------------------------
# Benchmark._configure_task_question — singleton state reset.
#
# `TaskMetadata.get_task` is a class-level cache, so the same task object is
# reused across benchmark runs. `_configure_task_question` sets all Q&A axes
# unconditionally, so state from a prior run (e.g. a CoT generated-text cell)
# cannot leak into a later plain multiple-choice cell.
# ----------------------------------------------------------------------


@pytest.fixture
def fresh_task() -> TaskMetadata:
    """A real `TaskMetadata` with both MC and numeric Q&A interfaces wired up,
    so `_configure_task_question` can be exercised without monkeypatching the
    singleton cache. Uses a unique name to avoid colliding with the cache."""
    mc_qa = MultipleChoiceQA(
        column="TARGET",
        text="Is the value high?",
        choices=(
            Choice("low", data_value=0, numeric_value=0.0),
            Choice("high", data_value=1, numeric_value=1.0),
        ),
    )
    num_qa = DirectNumericQA(column="TARGET", text="What is the value?")
    name = f"_test_state_leak_{id(mc_qa)}"
    task = TaskMetadata(
        name=name,
        features=["x"],
        target="TARGET",
        cols_to_text={
            "x": ColumnToText("x", short_description="x"),
            "TARGET": ColumnToText("TARGET", short_description="target"),
        },
        multiple_choice_qa=mc_qa,
        direct_numeric_qa=num_qa,
    )
    yield task
    TaskMetadata._tasks.pop(name, None)


def _cot_numeric_config(**extra) -> BenchmarkConfig:
    """Upstream's ``ChainOfThoughtQA`` == our ``TextNumericQA``: a numeric answer
    decoded from generated text, with the CoT instruction toggled on. Unlike
    upstream (where CoT was its own non-numeric mode), here CoT *is* numeric +
    generated-text, so `_use_numeric_qa` is True for these configs."""
    return BenchmarkConfig(
        numeric_risk_prompting=True,
        use_generated_text=True,
        cot_prompting=True,
        **extra,
    )


class TestConfigureTaskQuestionStateReset:
    def test_chat_mcq_after_cot_thinking_resets_to_mc(self, fresh_task):
        # Configure CoT + thinking first, then plain chat MC. Because
        # _configure_task_question sets every Q&A axis unconditionally, the
        # leaked TextNumericQA cannot survive into the MC cell.
        Benchmark._configure_task_question(fresh_task, _cot_numeric_config(reasoning="high"))
        assert isinstance(fresh_task.question, TextNumericQA)
        assert fresh_task._cot_prompting is True
        assert fresh_task._use_numeric_qa is True  # our CoT is numeric + generated-text

        Benchmark._configure_task_question(fresh_task, BenchmarkConfig(use_chat_template=True))
        assert isinstance(fresh_task.question, MultipleChoiceQA)
        assert not isinstance(fresh_task.question, TextNumericQA)
        assert fresh_task._cot_prompting is False
        assert fresh_task._use_numeric_qa is False
        assert fresh_task._use_generated_text_for_qa is False

    def test_chat_mcq_after_numeric_resets_to_mc(self, fresh_task):
        # Symmetric case: numeric (token-prob) -> chat MC must also clear state.
        Benchmark._configure_task_question(fresh_task, BenchmarkConfig(numeric_risk_prompting=True))
        assert isinstance(fresh_task.question, DirectNumericQA)
        assert not isinstance(fresh_task.question, TextNumericQA)
        assert fresh_task._use_numeric_qa is True

        Benchmark._configure_task_question(fresh_task, BenchmarkConfig(use_chat_template=True))
        assert isinstance(fresh_task.question, MultipleChoiceQA)
        assert fresh_task._cot_prompting is False
        assert fresh_task._use_numeric_qa is False

    def test_chat_mcq_after_cot_resets_to_mc(self, fresh_task):
        # CoT without thinking -- same leak pattern.
        Benchmark._configure_task_question(fresh_task, _cot_numeric_config())
        assert isinstance(fresh_task.question, TextNumericQA)

        Benchmark._configure_task_question(fresh_task, BenchmarkConfig())  # all-default = plain MC
        assert isinstance(fresh_task.question, MultipleChoiceQA)
        assert not isinstance(fresh_task.question, TextNumericQA)

    def test_cot_after_numeric_overrides(self, fresh_task):
        # numeric (token-prob) -> CoT (generated-text) must switch the type.
        Benchmark._configure_task_question(fresh_task, BenchmarkConfig(numeric_risk_prompting=True))
        Benchmark._configure_task_question(fresh_task, _cot_numeric_config())
        assert isinstance(fresh_task.question, TextNumericQA)
        # Inverted vs upstream: our CoT stays numeric (TextNumericQA <: DirectNumericQA).
        assert fresh_task._use_numeric_qa is True
        assert fresh_task._cot_prompting is True

    def test_numeric_after_cot_thinking_overrides(self, fresh_task):
        # CoT + thinking -> numeric (token-prob) must clear the CoT/text flags.
        Benchmark._configure_task_question(fresh_task, _cot_numeric_config(reasoning="high"))
        Benchmark._configure_task_question(fresh_task, BenchmarkConfig(numeric_risk_prompting=True))
        assert isinstance(fresh_task.question, DirectNumericQA)
        assert not isinstance(fresh_task.question, TextNumericQA)
        assert fresh_task._cot_prompting is False
        assert fresh_task._use_numeric_qa is True
        assert fresh_task._use_generated_text_for_qa is False
