"""Module for using a language model through a web API for risk classification."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from litellm import ModelResponse
    from litellm.types.llms.openai import ResponsesAPIResponse


import dotenv
import numpy as np

from folktexts.llm_utils import decode_topk_logprobs_to_risk_estimate, get_model_developer
from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata
from folktexts.token_tracker import TokenTracker

from .base import EncodeRowCallable, LLMClassifier


@dataclass
class _ModelConfig:
    """Configuration for a single web-API model."""

    azure_api_version: str
    deployment_name: str | None = None  # None = use model name as-is
    max_tpm: int = 0
    max_rpm: int = 0
    is_reasoning_model: bool = False


# Registry of known web-API models. Add new models here — one entry covers all
# properties. Can be extended at runtime via WebAPILLMClassifier.load_model_registry().
_MODEL_REGISTRY: dict[str, _ModelConfig] = {
    "claude-opus-4-5": _ModelConfig(
        azure_api_version="20251101",
        deployment_name="azure_ai/claude-opus-4-5",
        max_tpm=10000,
        max_rpm=10,
        is_reasoning_model=True,
    ),
    "o3": _ModelConfig(
        azure_api_version="2025-04-16",
        max_tpm=250000,
        max_rpm=250,
        is_reasoning_model=True,
    ),
    "o3-mini": _ModelConfig(
        azure_api_version="2025-01-31",
        max_tpm=2100000,
        max_rpm=210,
        is_reasoning_model=True,
    ),
    "gpt-5.1": _ModelConfig(
        azure_api_version="2025-11-13",
        max_tpm=50000,
        max_rpm=500,
    ),
    "gpt-5.2": _ModelConfig(
        azure_api_version="unknown",
        max_tpm=50000,
        max_rpm=500,
        is_reasoning_model=True,
    ),
    "gpt-5.4": _ModelConfig(
        azure_api_version="unknown",
        max_tpm=50000,
        max_rpm=500,
        is_reasoning_model=True,
    ),
    "DeepSeek-V3.2": _ModelConfig(
        azure_api_version="1",
        deployment_name="openai/DeepSeek-V3.2",
        max_tpm=5000000,
        max_rpm=5000,
    ),
    "DeepSeek-R1": _ModelConfig(
        azure_api_version="1",
        deployment_name="openai/DeepSeek-R1",
        max_tpm=5000000,
        max_rpm=5000,
        is_reasoning_model=True,
    ),
    "gpt-4.1": _ModelConfig(
        azure_api_version="2025-04-14",
        max_tpm=110000,
        max_rpm=110,
    ),
    "o1": _ModelConfig(
        azure_api_version="2024-12-17",
        max_tpm=780000,
        max_rpm=130,
        is_reasoning_model=True,
    ),
    "gpt-4o-mini": _ModelConfig(
        azure_api_version="2024-07-18",
        max_tpm=250000,
        max_rpm=2500,
    ),
    "o4-mini": _ModelConfig(
        azure_api_version="2025-04-16",
        max_tpm=250000,
        max_rpm=250,
        is_reasoning_model=True,
    ),
    "Kimi-K2-Thinking": _ModelConfig(
        azure_api_version="1",
        deployment_name="openai/Kimi-K2-Thinking",
        is_reasoning_model=True,
    ),
    "Kimi-K2.5": _ModelConfig(
        azure_api_version="1",
        deployment_name="openai/Kimi-K2.5",
        is_reasoning_model=True,
    ),
    "grok-4-fast-reasoning": _ModelConfig(
        azure_api_version="1",
        deployment_name="openai/grok-4-fast-reasoning",
        is_reasoning_model=True,
    ),
    "grok-4-fast-non-reasoning": _ModelConfig(
        azure_api_version="1",
        deployment_name="openai/grok-4-fast-non-reasoning",
    ),
}


_DEFAULT_MAX_RPM: int = min(cfg.max_rpm for cfg in _MODEL_REGISTRY.values() if cfg.max_rpm)
_DEFAULT_MAX_TPM: int = min(cfg.max_tpm for cfg in _MODEL_REGISTRY.values() if cfg.max_tpm)


class WebAPILLMClassifier(LLMClassifier):
    """Use an LLM through a web API to produce risk scores."""

    _registry: dict[str, _ModelConfig] = _MODEL_REGISTRY

    def __init__(
        self,
        model_name: str,
        task: TaskMetadata | str,
        encode_row: EncodeRowCallable = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        max_api_rpm: int = _DEFAULT_MAX_RPM,
        max_api_tpm: int = _DEFAULT_MAX_TPM,
        seed: int = 42,
        token_tracker: TokenTracker | None = None,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier object that uses a web API for inference.

        Parameters
        ----------
        model_name : str
            The model ID to be resolved by `litellm`.
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
        max_api_rpm : int, optional
            The maximum number of requests per minute allowed for the API.
        seed : int, optional
            The random seed - used for reproducibility.
        **inference_kwargs
            Additional keyword arguments to be used at inference time. Options
            include `context_size` and `batch_size`.
        """
        dotenv_success = dotenv.load_dotenv()
        logging.debug(f"dotenv.load_dotenv() returned {dotenv_success}.")

        super().__init__(
            model_name=model_name,
            task=task,
            encode_row=encode_row,
            threshold=threshold,
            correct_order_bias=correct_order_bias,
            seed=seed,
            **inference_kwargs,
        )
        model_cfg = self._registry.get(model_name)
        self.deployment_name = (model_cfg.deployment_name or model_name) if model_cfg else model_name

        # Initialize total cost of API calls
        self._total_cost = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._num_api_calls = 0
        self._total_response_time = 0.0

        # Optional token tracker; previous cumulative totals used to compute per-batch deltas
        self.token_tracker = token_tracker
        self._prev_tracker_prompt_tokens = 0
        self._prev_tracker_completion_tokens = 0

        # Set maximum requests / tokens per minute (env vars take priority over defaults)
        self.max_api_rpm = max(max_api_rpm, model_cfg.max_rpm if model_cfg else 0)
        if rpm_env := os.getenv("MAX_API_RPM"):
            logging.info(f"MAX_API_RPM env var overrides previous value of{self.max_api_rpm} with {rpm_env}.")
            self.max_api_rpm = int(rpm_env)

        self.max_api_tpm = max(max_api_tpm, model_cfg.max_tpm if model_cfg else 0)
        if tpm_env := os.getenv("MAX_API_TPM"):
            logging.warning(f"MAX_API_TPM env var overrides previous value of {self.max_api_tpm} with {tpm_env}.")
            self.max_api_tpm = int(tpm_env)

        # Check extra dependencies
        assert self.check_webAPI_deps(), "Web API dependencies are not installed."

        # Check the API key for the endpoint this model actually routes to.
        # All registry models are Azure-routed: `azure_ai/*` deployments use the
        # Azure AI inference endpoint (AZURE_AI_API_KEY), everything else uses
        # Azure OpenAI (AZURE_API_KEY).
        if self.deployment_name.startswith("azure_ai"):
            if "AZURE_AI_API_KEY" not in os.environ:
                raise ValueError("AZURE_AI_API_KEY not found in environment variables")
            if "AZURE_AI_API_BASE" not in os.environ:
                raise ValueError("AZURE_AI_API_BASE not found in environment variables")
        else:
            if "AZURE_API_KEY" not in os.environ:
                raise ValueError("AZURE_API_KEY not found in environment variables")
            if "AZURE_API_BASE" not in os.environ:
                raise ValueError("AZURE_API_BASE not found in environment variables")

        # Validate reasoning argument for known reasoning models
        reasoning = self.inference_kwargs.get("reasoning")
        if (model_cfg and model_cfg.is_reasoning_model) and reasoning is None:
            raise ValueError(
                f"Model '{self.model_name}' is a reasoning model — please specify --reasoning (e.g. 'medium', '0.25', '0')."
            )

        # Set API type
        self.api_type = "completion"

        # litellm completion does not provide reasoning with opt-in summary -> switch to responses API
        if (
            get_model_developer(self.model_name) == "OpenAI"
            and reasoning is not None
            and self.task.question.use_generated_text
        ):
            # log-probs not available via responses API, but then reasoning can only be a str!
            logging.debug(
                f"Using responses API for OpenAI reasoning model {self.model_name} "
                "with thinking enabled and text-based extraction."
            )

            self.api_type = "responses"

        # Set-up litellm API client
        import litellm

        litellm.success_callback = [self.track_cost_callback]

        # from litellm import completion, responses

        # Get supported parameters
        from litellm import get_supported_openai_params

        supported_params = get_supported_openai_params(model=self.deployment_name)
        if self.api_type == "responses":
            from litellm import OpenAIResponsesAPIConfig

            config = OpenAIResponsesAPIConfig()
            # merge lists of suppported parameters (parameters for the completion API should get
            # mapped internally by the response API)
            supported_params = list(
                set(supported_params or []) | set(config.get_supported_openai_params(model=self.deployment_name) or [])
            )

        if supported_params is None:
            raise RuntimeError(f"Failed to get supported parameters for model '{self.deployment_name}'.")
        self.supported_params = set(supported_params)
        self._warned_unsupported_params: set[str] = set()

        # Set litellm logger level to WARNING
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        # Create rate-limited API client
        from llm_api_client import APIClient

        self.client = APIClient(
            max_requests_per_minute=self.max_api_rpm,
            max_tokens_per_minute=self.max_api_tpm,
            api_type=self.api_type,
        )

    @staticmethod
    def check_webAPI_deps() -> bool:
        """Check if litellm dependencies are available."""
        try:
            import litellm  # noqa: F401
            import llm_api_client  # noqa: F401
        except ImportError:
            logging.critical(
                "Please install extra API dependencies with `pip install 'folktexts[apis]'` to use the WebAPILLMClassifier."
            )
            return False
        return True

    @classmethod
    def load_model_registry(cls, path: str) -> None:
        """Merge additional model configs from a YAML file into the registry.

        The YAML file should be a mapping of model names to config fields, e.g.::

            my-new-model:
                azure_api_version: "2025-01-01"
                deployment_name: "openai/my-new-model"
                max_tpm: 100000
                max_rpm: 500
                is_reasoning_model: false
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("PyYAML is required to load a model registry file. Install it with: pip install pyyaml")

        with open(path) as f:
            data = yaml.safe_load(f)
        for model_name, fields in data.items():
            cls._registry[model_name] = _ModelConfig(**fields)
        logging.info(f"Loaded {len(data)} model config(s) from '{path}'.")

    def _filter_supported_params(self, params: dict) -> dict:
        """Drop params the model's API doesn't support, warning about each drop.

        Web APIs (notably OpenAI reasoning models such as o1/o3) reject
        parameters like `temperature`; filtering keeps the request valid
        instead of raising, while the warning keeps the drop visible.
        """
        unsupported = [k for k in params if k not in self.supported_params]
        if set(unsupported) - self._warned_unsupported_params:  # warn once per param
            self._warned_unsupported_params.update(unsupported)
            logging.warning(
                f"Model '{self.model_name}' does not support API "
                f"parameter(s) {sorted(unsupported)}; dropping them from the "
                f"request (this may reduce determinism/reproducibility). "
                f"Supported params: {sorted(self.supported_params)}."
            )
        return {k: v for k, v in params.items() if k in self.supported_params}

    def _query_webapi_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> list[ModelResponse]:
        """Query the web API with a batch of prompts and returns the json response.

        TODO! Retry on non-successful API calls (e.g., RPM exceeded).

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
        responses_batch : list[ModelResponse]
            The returned API responses for each prompt in the batch.
        """
        # TODO
        # Handle longer text generation
        # if isinstance(question, ChainOfThoughtQA):
        #     api_call_params = dict(
        #         temperature=self._resolve_temperature(question),
        #         max_tokens=question.max_new_tokens,
        #         stream=False,
        #         seed=self.seed,
        #     )
        #     # Use the user-supplied system prompt (via PromptConfig / --system-prompt)
        #     # when set; otherwise fall back to the default CoT instruction.
        #     if self.prompt_config.system_prompt is not None:
        #         system_prompt = self.prompt_config.system_prompt()
        #     else:
        #         system_prompt = (
        #             "You are a helpful assistant. Reason step-by-step about the question "
        #             "and provide your final probability estimate. Your response MUST end "
        #             "with 'Probability: X%' where X is a number between 0 and 100."
        #         )

        # Adapt number of forward passes for token-probability based methods.
        if question.num_forward_passes == 1:
            # Single token answers should require only one forward pass
            num_forward_passes = 1
        else:
            # NOTE: Models often generate "0." instead of directly outputting the fractional part
            # > Therefore: for multi-token answers, extra forward passes may be required
            # Add extra tokens for textual prefix, e.g., "The probability is: ..."
            num_forward_passes = question.num_forward_passes + 2

        # Build API parameter calls for the two decoding paths
        # Free-form text generation: sample a full response and parse the answer from the generated text
        if question.use_generated_text:
            api_call_params = dict(
                temperature=self._resolve_temperature(question),  # 1
                max_completion_tokens=self.inference_kwargs["max_new_tokens"],  # question.max_new_tokens,
                stream=False,
                seed=self.seed,
            )
        else:
            # Token-probability decoding: read top_logprobs of the answer tokens
            # Temperature is always 0 here: MC/numeric decode from the returned
            # top_logprobs (which OpenAI-style APIs report untempered), so sampling
            # would only add noise to the multi-pass token trajectory.
            api_call_params = dict(
                temperature=0,
                max_tokens=num_forward_passes,  # max(num_forward_passes, self.inference_kwargs.get("max_new_tokens", 0)),
                stream=False,
                seed=self.seed,
                logprobs=True,
                top_logprobs=20,
            )

        if self.model_name.startswith("claude"):
            api_call_params.pop("seed")
            logging.debug("Removed 'seed' from API call parameters for Claude model, as it is not supported.")

        # Set extra arguments for reasoning-augmented models
        _OPENAI_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "auto")
        reasoning = self.inference_kwargs.get("reasoning")
        logging.debug(f"reasoning is set to: {reasoning}")
        if reasoning is not None and reasoning != "0":
            if self.model_name.startswith("claude"):
                # Claude models allow to pass reasoning budget either as number of tokens or as fraction of max_new_tokens
                # minimum number of tokens is restricted to 1024
                val = float(reasoning)
                max_new_tokens = self.inference_kwargs["max_new_tokens"]
                budget_tokens = int(val * max_new_tokens) if val <= 1.0 else int(val)
                budget_tokens = max(budget_tokens, 1024)
                assert budget_tokens <= max_new_tokens, (
                    f"budget_tokens ({budget_tokens}) must not exceed max_new_tokens ({max_new_tokens})"
                )
                logging.info(f"Thinking enabled for Claude model with budget_tokens={budget_tokens}.")
                api_call_params["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
            elif get_model_developer(self.model_name) == "OpenAI":
                # NOTE: reasoning_effort accepts: "none", "minimal", "low", "medium", "high", "xhigh"
                # https://docs.litellm.ai/docs/providers/openai
                if reasoning not in _OPENAI_EFFORT_LEVELS:
                    raise ValueError(
                        f"Invalid reasoning effort '{reasoning}' for OpenAI model. Must be one of: {_OPENAI_EFFORT_LEVELS}"
                    )
                logging.info(f"Thinking enabled for OpenAI model with reasoning_effort='{reasoning}'.")
                if self.api_type == "responses":
                    # summary only available via responses API, but that does not support logprobs
                    # -> only use if extracting answer from generated text
                    api_call_params["reasoning"] = {
                        "effort": reasoning,
                        "summary": "detailed",
                    }
                else:
                    api_call_params["reasoning_effort"] = reasoning

        api_call_params = self._filter_supported_params(api_call_params)

        # `logprobs` are load-bearing for token-probability decoding: dropping
        # them would only fail later, deep inside response decoding. Fail fast
        # instead (e.g. OpenAI o1/o3 don't support logprobs). The generated-text
        # path parses the answer from text and intentionally omits logprobs.
        if not question.use_generated_text and "logprobs" not in api_call_params:
            raise RuntimeError(
                f"Model '{self.model_name}' does not support `logprobs`, which "
                f"are required to decode multiple-choice/numeric risk estimates. "
                f"Use freefrom text generation instead."
            )

        # Get system prompt depending on Q&A type
        if isinstance(question, DirectNumericQA) or isinstance(question, MultipleChoiceQA):
            # Use the system prompt carried by PromptConfig (always the QA subclass
            # default unless the caller explicitly cleared it). `None` disables the
            # system role entirely. Bind unconditionally so it is always defined.
            system_prompt = self.prompt_config.system_prompt() if self.prompt_config.system_prompt is not None else None
        else:
            raise ValueError(f"Unknown question type '{type(question)}'.")
        logging.debug(f"System prompt: {system_prompt}")

        # Query model for each prompt in the batch
        requests_data = []
        for prompt in prompts_batch:
            # Construct prompt messages (omit the system role when disabled)
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            requests_data.append(
                {
                    "model": self.deployment_name,
                    "api_base": (
                        os.environ["AZURE_AI_API_BASE"]
                        if self.deployment_name.startswith("azure_ai")
                        else os.environ["AZURE_API_BASE"]
                    ),
                    "messages": messages,
                    **api_call_params,
                }
            )

        logging.debug(f"API call parameters: {api_call_params}")
        logging.debug(f"First request data: {requests_data[0]}")

        if self.api_type == "responses":
            for p in range(len(prompts_batch)):
                requests_data[p]["input"] = requests_data[p].pop("messages")
        responses_batch = self.client.make_requests_with_retries(requests_data, max_retries=10, sanitize=False)

        return responses_batch

    @staticmethod
    def _extract_reasoning_tokens(response: ModelResponse | ResponsesAPIResponse) -> int | None:
        """Extract the number of reasoning tokens used from an API response.

        The field lives under a different attribute depending on the backend:
        the responses API (OpenAI reasoning models) exposes
        ``usage.output_tokens_details.reasoning_tokens`` while the completion API
        (Claude, DeepSeek, Kimi, etc.) exposes
        ``usage.completion_tokens_details.reasoning_tokens``. Returns ``None`` if
        the information is not available.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        for details_attr in ("output_tokens_details", "completion_tokens_details"):
            details = getattr(usage, details_attr, None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
                if reasoning_tokens is not None:
                    return reasoning_tokens
        return None

    def _decode_risk_estimate_from_api_response(
        self,
        response: ModelResponse | ResponsesAPIResponse,
        question: MultipleChoiceQA | DirectNumericQA,
    ) -> tuple[float, Any]:
        """Decode model output from API response to get risk estimate.

        Parameters
        ----------
        response : ModelResponse | ResponsesAPIResponse
            The response from the API call.
        question : MultipleChoiceQA | DirectNumericQA
            The question (`QAInterface`) object to use for querying the model.

        Returns
        -------
        risk_estimate : float
            The risk estimate for the API query.
        extra : Any
            Additional output metadata (reasoning content, raw response, token probs, etc.).
        """

        if question.use_generated_text:
            reasoning_content = ""

            if self.api_type == "completion":
                choice = response.choices[0]
                response_message = choice.message.content
                if "reasoning_content" in choice.message.__dict__.keys():
                    # applicable for DepSeek, Claude, Kimi
                    reasoning_content = choice.message.reasoning_content
                    logging.debug(f"Received reasoning content: {reasoning_content}")

                if self.inference_kwargs.get("reasoning") not in (None, "0") and len(reasoning_content) == 0:
                    logging.debug("Reasoning enabled, but no reasoning content found in response.")
            else:
                # response API
                output_texts = []
                for item in response.output:
                    if item.type == "message":
                        for content in item.content:
                            if content.type == "output_text":
                                output_texts.append(content.text)
                    if item.type == "reasoning":
                        if hasattr(item, "summary") and item.summary:
                            for summary in item.summary:
                                reasoning_content += summary.text + "\n"

                response_message = "\n".join(output_texts) if len(output_texts) > 0 else ""
                if self.inference_kwargs.get("reasoning") not in (None, "0"):
                    if len(reasoning_content) == 0:
                        logging.debug("Reasoning enabled, but no summary returned.")
                    else:
                        logging.debug(f"Received reasoning summary (length={len(reasoning_content)}).")
                elif len(reasoning_content) > 0:
                    logging.debug(f"Reasoning not enabled, but received reasoning content: {reasoning_content}")

            if response_message is None:
                logging.warning("No response message API response, setting it to empty string.")
                response_message = ""
            else:
                logging.debug(f"Received response_message: {response_message}")
            risk_estimate = question.get_answer_from_model_output(
                text=response_message,
            )
            reasoning_tokens = self._extract_reasoning_tokens(response)
            return (
                risk_estimate,
                {
                    "reasoning": reasoning_content,
                    "response": response_message,
                    "reasoning_tokens": reasoning_tokens,
                },
            )

        else:
            assert self.api_type != "responses", "Logprobs not available via responses API."
            # Get response message
            choice = response.choices[0]
            response_message = choice.message.content
            logging.debug(f"Received response_message: {response_message}")

            # Get top-K logprobs per forward pass (keyed by decoded token string).
            # OpenAI-style API returns string keys; we synthesise an integer ID per
            # unique string so we can share the same scatter/decode helper as the
            # vLLM backend (which provides real token IDs directly).
            assert choice.logprobs is not None
            token_choices_all_passes = choice.logprobs.content
            assert token_choices_all_passes is not None

            token_logprobs_per_pass = [
                {token_metadata.token: token_metadata.logprob for token_metadata in top_token_logprobs.top_logprobs}
                for top_token_logprobs in token_choices_all_passes
            ]

            # Decode model output into risk estimates
            # 1. Construct vocabulary dict for this response
            all_tokens = sorted({tok for d in token_logprobs_per_pass for tok in d})
            synthetic_vocab = {tok: idx for idx, tok in enumerate(all_tokens)}

            # 2. Parse `token_logprobs_per_pass` into a list of num_passes dicts
            #    mapping synthetic token ID to log probability
            per_pass_topk = [
                {synthetic_vocab[tok]: lp for tok, lp in pass_logprobs.items()} for pass_logprobs in token_logprobs_per_pass
            ]

            # Get risk estimate
            risk_estimate = decode_topk_logprobs_to_risk_estimate(
                per_pass_topk,
                tokenizer_vocab=synthetic_vocab,
                vocab_dim=len(synthetic_vocab),
                question=question,
            )

            # Sanity check numeric answers based on global model response:
            if isinstance(question, DirectNumericQA):
                try:
                    _match = re.match(r"[-+]?\d*\.\d+|\d+", response_message)
                    if _match is None:
                        raise ValueError(f"No numeric token found in '{response_message}'")
                    numeric_response = _match.group()
                    risk_estimate_full_text = float(numeric_response)

                    if not np.isclose(risk_estimate, risk_estimate_full_text, atol=1e-2):
                        logging.info(
                            f"Numeric answer mismatch: {risk_estimate} != {risk_estimate_full_text} "
                            f"from response '{response_message}'."
                        )

                        # Using full text answer as it more tightly relates to the ChatGPT web answer
                        risk_estimate = risk_estimate_full_text

                        if risk_estimate > 1:
                            logging.info(
                                f"Got risk estimate > 1: {risk_estimate}. Using "
                                f"output as a percentage: {risk_estimate / 100.0} instead."
                            )
                            risk_estimate = risk_estimate / 100.0

                except Exception:
                    logging.info(
                        f"Failed to extract numeric response from message='{response_message}';\n"
                        f"Falling back on standard risk estimate of {risk_estimate}."
                    )

            return risk_estimate, token_logprobs_per_pass
            # TODO: consider passing risk_estimate, None - as logprobs carry no additional metadata, and not used downstream

    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> tuple[list[float], list[Any]]:
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
        risk_estimates : list[float]
            The risk estimates for each prompt in the batch.
        outputs : list[Any]
            Additional output metadata for each prompt.

        Raises
        ------
        RuntimeError
            Raised when web API call is unsuccessful.
        """

        # Query model through web API
        api_responses_batch = self._query_webapi_batch(
            prompts_batch=prompts_batch,
            question=question,
            context_size=context_size,
        )

        # Parse API responses and decode model output
        risk_estimates_batch: list = []
        outputs_batch: list = []
        for i, response in enumerate(api_responses_batch):
            if not response:
                logging.error(f"Response {i + 1}: Request failed (empty response). Adding NaN value.")
                risk_estimates_batch.append(np.nan)
                outputs_batch.append(None)
                continue
            try:
                if self.api_type == "completion":
                    message_content = response.choices[0].message.content
                else:
                    # Responses API
                    message_content = ""
                    for item in response.output:
                        if item.type == "message":
                            for content in item.content:
                                if content.type == "output_text":
                                    message_content += content.text + "\n"
                if message_content is not None:
                    logging.debug(f"Response {i + 1}: {message_content[:100]}...")  # Print first 100 chars
                else:
                    logging.debug(f"Response {i + 1} is None.")
                risk_est, out = self._decode_risk_estimate_from_api_response(response, question)
                risk_estimates_batch.append(risk_est)
                outputs_batch.append(out)
            except (AttributeError, IndexError, TypeError, AssertionError, ValueError) as e:
                logging.error(f"Response {i + 1}: Could not parse response content. Error: {e}")
                logging.error(f"Raw response: {response}")
                logging.error("Adding NaN value.")
                risk_estimates_batch.append(np.nan)
                outputs_batch.append(None)

        self.track_stats(batch_size=len(prompts_batch))
        return np.asarray(risk_estimates_batch), outputs_batch

    # def track_cost_callback(
    #     self,
    #     kwargs,
    #     completion_response,
    #     start_time,
    #     end_time,
    # ):
    #     """Callback function to cost of API calls."""
    #     try:
    #         response_cost = kwargs.get("response_cost", 0)
    #         self._total_cost += response_cost

    #     except Exception as e:
    #         logging.error(f"Failed to track cost of API calls: {e}")

    def track_stats(self, batch_size: int = 1):
        # get all tracker attributes with defaults
        total_cost = getattr(self.client.tracker, "total_cost", 0)
        total_prompt_tokens = getattr(self.client.tracker, "total_prompt_tokens", 0)
        total_completion_tokens = getattr(self.client.tracker, "total_completion_tokens", 0)
        num_api_calls = getattr(self.client.tracker, "num_api_calls", 0)
        mean_response_time = getattr(self.client.tracker, "mean_response_time", None)

        logging.info(f"Total cost: ${total_cost:.4f}")
        logging.info(f"Total prompt tokens: {total_prompt_tokens}")
        logging.info(f"Total completion tokens: {total_completion_tokens}")
        logging.info(f"Number of successful API calls: {num_api_calls}")

        if mean_response_time is not None and isinstance(mean_response_time, (int, float)):
            logging.info(f"Mean response time: {mean_response_time:.2f}s")
        else:
            logging.info("Mean response time: Not available")

        if self.token_tracker is not None:
            batch_prompt = total_prompt_tokens - self._prev_tracker_prompt_tokens
            batch_completion = total_completion_tokens - self._prev_tracker_completion_tokens
            if batch_prompt > 0 or batch_completion > 0:
                self.token_tracker.record_batch(
                    prompt_tokens=batch_prompt,
                    completion_tokens=batch_completion,
                    batch_size=batch_size,
                )
            self._prev_tracker_prompt_tokens = total_prompt_tokens
            self._prev_tracker_completion_tokens = total_completion_tokens

    def track_cost_callback(
        self,
        kwargs,
        completion_response,
        start_time,
        end_time,
    ):
        """Callback function to track cost of API calls."""
        try:
            # Extract cost properly
            response_cost = 0

            # Try different ways to get cost
            if "response_cost" in kwargs:
                response_cost = kwargs["response_cost"]
            elif hasattr(completion_response, "cost"):
                response_cost = getattr(completion_response, "cost", 0)
            elif isinstance(completion_response, dict) and "cost" in completion_response:
                response_cost = completion_response["cost"]

            # Update total cost
            self._total_cost += response_cost

            # Update tracker if it has update_stats method
            if hasattr(self.client.tracker, "update_stats"):
                self.client.tracker.update_stats(
                    response_cost=response_cost,
                    start_time=start_time,
                    end_time=end_time,
                )
            elif hasattr(self.client.tracker, "add_cost"):
                self.client.tracker.add_cost(response_cost)

        except Exception as e:
            logging.error(f"Failed to track cost of API calls: {e}")
            logging.exception("Full traceback:")

    def __del__(self):
        """Destructor to report total cost of API calls."""
        msg = f"Total cost of API calls: ${self._total_cost:.2f}"
        print(msg)
        logging.info(msg)
