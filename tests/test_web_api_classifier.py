"""Regression tests for ``WebAPILLMClassifier._query_webapi_batch``.

These exercise the request-building logic without a live web API: the classifier
is built with ``__new__`` (bypassing the litellm / API-key / ``APIClient`` setup
in ``__init__``), and the rate-limited network seam
(``self.client.make_requests_with_retries``) is replaced by a stub that records
the ``requests_data`` payload and returns canned responses. Each recorded
request dict carries the ``messages`` (or ``input`` for the responses API) plus
the merged ``api_call_params`` (temperature, logprobs, …).

The key regression: for numeric/MCQ questions, ``system_prompt`` must always be
bound. Before the fix it was assigned only inside
``if self.prompt_config.system_prompt is not None``, so a config with the
system role explicitly disabled (``system_prompt=None``) raised
``NameError: name 'system_prompt' is not defined``.
"""

from __future__ import annotations

import pytest
from folktexts.classifier import WebAPILLMClassifier
from folktexts.classifier.base import InferenceConfig
from folktexts.prompting import PromptConfig


@pytest.fixture(autouse=True)
def _fake_api_base(monkeypatch):
    """`_query_webapi_batch` reads the API base from the environment."""
    monkeypatch.setenv("AZURE_API_BASE", "https://test.invalid")
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://test.invalid")


@pytest.fixture(scope="module")
def mcq_task():
    from folktexts.acs import ACSTaskMetadata

    return ACSTaskMetadata.get_task("ACSIncome", use_numeric_qa=False)


def _make_classifier(
    prompt_config: PromptConfig,
    *,
    supported_params: set | None = None,
    api_type: str = "completion",
):
    """Build a WebAPILLMClassifier without touching litellm / the network.

    Returns ``(clf, calls)`` where ``calls`` accumulates the per-request message
    lists (one per prompt); the resolved ``api_call_params`` for the last request
    are exposed on ``clf.last_call_params``.
    """
    clf = WebAPILLMClassifier.__new__(WebAPILLMClassifier)
    clf._model_name = "test-model"
    clf.deployment_name = "test-model"
    clf.api_type = api_type
    clf._prompt_config = prompt_config
    # No temperature override → defer to each question's default_temperature.
    clf._inference = InferenceConfig(seed=42, temperature=None)
    clf.supported_params = (
        supported_params
        if supported_params is not None
        else {
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "seed",
            "logprobs",
            "top_logprobs",
        }
    )
    clf._warned_unsupported_params = set()

    calls: list[list[dict]] = []

    def _messages(req: dict) -> list[dict]:
        # The responses API renames `messages` -> `input`.
        return req.get("messages", req.get("input"))

    class _StubClient:
        """Stand-in for the rate-limited ``APIClient``."""

        def make_requests_with_retries(self, requests_data, **kwargs):
            for req in requests_data:
                calls.append(_messages(req))
            clf.last_request = requests_data[0]  # raw request dict (routing + params)
            # `api_call_params` are merged into each request dict alongside the
            # routing keys; expose them for the param-contract assertions.
            routing = {"model", "api_base", "messages", "input"}
            clf.last_call_params = {k: v for k, v in requests_data[0].items() if k not in routing}
            return [{"choices": [{"message": {"content": "Probability: 50%"}}]} for _ in requests_data]

    clf.client = _StubClient()
    return clf, calls


def _system_contents(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "system"]


# --- MCQ / Numeric -------------------------------------------------


def test_mcq_with_disabled_system_prompt_does_not_raise(mcq_task):
    """Regression: system_prompt=None previously raised NameError on MCQ."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task, system_prompt=None)
    assert cfg.system_prompt is None  # precondition: role disabled

    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["some prompt"], question=mcq_task.multiple_choice_qa)

    assert len(calls) == 1
    # System role is omitted entirely when disabled (no content=None turn).
    assert _system_contents(calls[0]) == []
    assert any(m["role"] == "user" for m in calls[0])


def test_numeric_with_disabled_system_prompt_does_not_raise(mcq_task):
    cfg = PromptConfig.from_dict(
        pv={},
        task=mcq_task,
        question=mcq_task.direct_numeric_qa,
        system_prompt=None,
    )
    assert cfg.system_prompt is None

    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.direct_numeric_qa)

    assert _system_contents(calls[0]) == []


def test_mcq_default_system_prompt_is_sent(mcq_task):
    """Default config carries the MCQ system prompt → emitted as the system turn."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)  # PROMPT_DEFAULT
    assert cfg.system_prompt is not None

    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)

    sys_contents = _system_contents(calls[0])
    assert sys_contents == [cfg.system_prompt()]
    assert sys_contents[0]  # non-empty


def test_custom_system_prompt_overrides_default(mcq_task):
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task, system_prompt="CUSTOM SYS")
    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert _system_contents(calls[0]) == ["CUSTOM SYS"]


# --- Temperature contract ----------------------------------------------------


def test_mcq_uses_temperature_zero(mcq_task):
    """MCQ (token-probability) defaults to temperature 0 for determinism."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert clf.last_call_params["temperature"] == 0.0


def test_numeric_uses_temperature_zero(mcq_task):
    """Direct-numeric (token-probability) defaults to temperature 0."""
    cfg = PromptConfig.from_dict(
        pv={},
        task=mcq_task,
        question=mcq_task.direct_numeric_qa,
    )
    clf, _ = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.direct_numeric_qa)
    assert clf.last_call_params["temperature"] == 0.0


# --- Unsupported-parameter filtering -----------------------------------------


def test_unsupported_temperature_is_filtered_with_warning(mcq_task, caplog):
    """Models that reject `temperature` (e.g. o1/o3) must have it dropped, not raise."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    # Model supports everything EXCEPT temperature.
    clf, _ = _make_classifier(
        cfg,
        supported_params={"max_tokens", "stream", "seed", "logprobs", "top_logprobs"},
    )

    with caplog.at_level("WARNING"):
        clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)

    # temperature was dropped from the outgoing request instead of raising.
    assert "temperature" not in clf.last_call_params
    # The drop is visible in the logs and names the offending parameter.
    assert any("temperature" in rec.getMessage() and rec.levelname == "WARNING" for rec in caplog.records)


def test_all_supported_params_pass_through_unchanged(mcq_task, caplog):
    """No warning and no dropped keys when every param is supported."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)  # default supported_params covers all

    with caplog.at_level("WARNING"):
        clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)

    assert "temperature" in clf.last_call_params
    assert not any("does not support API parameter" in rec.getMessage() for rec in caplog.records)


def test_missing_logprobs_support_fails_fast_for_mcq(mcq_task):
    """`logprobs` is required to decode MCQ/numeric; dropping it must raise,
    not send a doomed request that only fails deep in response decoding."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, calls = _make_classifier(
        cfg,
        supported_params={"temperature", "max_tokens", "stream", "seed"},
    )

    with pytest.raises(RuntimeError, match="logprobs"):
        clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert calls == []  # no API call was made


def test_unsupported_param_warning_fires_once_across_batches(mcq_task, caplog):
    """The drop-warning must not spam the logs once per batch."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(
        cfg,
        supported_params={"max_tokens", "stream", "seed", "logprobs", "top_logprobs"},
    )

    with caplog.at_level("WARNING"):
        clf._query_webapi_batch(["p1"], question=mcq_task.multiple_choice_qa)
        clf._query_webapi_batch(["p2"], question=mcq_task.multiple_choice_qa)

    warnings = [rec for rec in caplog.records if "does not support API parameter" in rec.getMessage()]
    assert len(warnings) == 1
