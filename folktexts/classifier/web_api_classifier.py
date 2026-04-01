"""Module for using a language model through a web API for risk classification."""

from __future__ import annotations

import logging
import os
import re
from typing import Callable

import numpy as np
import pandas as pd

import dotenv

from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata

from .base import LLMClassifier
from folktexts.llm_utils import get_model_developer


class WebAPILLMClassifier(LLMClassifier):
    """Use an LLM through a web API to produce risk scores."""

    _model_to_azure_api_version = {
        "o3": "2025-04-16",
        "o3-mini": "2025-01-31",
        "gpt-5.1": "2025-11-13",
        "DeepSeek-V3.2": "1",
        "DeepSeek-R1": "1",
        "gpt-4.1": "2025-04-14",
        "o1": "2024-12-17",
        "gpt-4o-mini": "2024-07-18",
        "o4-mini": "2025-04-16",
        "claude-opus-4-5": "20251101",
        "Kimi-K2-Thinking": "1",
        "Kimi-K2.5": "1",
        "grok-4-fast-reasoning": "1",
        "grok-4-fast-non-reasoning": "1",
    }

    _model_to_azure_deployment_name = {
        "DeepSeek-V3.2": "openai/DeepSeek-V3.2",
        "DeepSeek-R1": "openai/DeepSeek-R1",
        "claude-opus-4-5": "azure_ai/claude-opus-4-5",
        "Kimi-K2-Thinking": "openai/Kimi-K2-Thinking",
        "Kimi-K2.5": "openai/Kimi-K2.5",
        "grok-4-fast-reasoning": "openai/grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning": "openai/grok-4-fast-non-reasoning",
    }

    _model_to_max_tpm = {
        "o3": 250000,
        "o3-mini": 2100000,
        "gpt-5.1": 50000,
        "DeepSeek-V3.2": 5000000,
        "DeepSeek-R1": 5000000,
        "gpt-4.1": 110000,
        "o1": 780000,
        "gpt-4o-mini": 250000,
        "o4-mini": 250000,
        "claude-opus-4-5": 10000,
    }

    _model_to_max_rpm = {
        "o3": 250,
        "o3-mini": 210,
        "gpt-5.1": 500,
        "DeepSeek-V3.2": 5000,
        "DeepSeek-R1": 5000,
        "gpt-4.1": 110,
        "o1": 130,
        "gpt-4o-mini": 2500,
        "o4-mini": 250,
        "claude-opus-4-5": 10,
    }

    def __init__(
        self,
        model_name: str,
        task: TaskMetadata | str,
        custom_prompt_prefix: str = None,
        encode_row: Callable[[pd.Series], str] = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        max_api_rpm: int = min(_model_to_max_rpm.values()),
        max_api_tpm: int = min(_model_to_max_tpm.values()),
        seed: int = 42,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier object that uses a web API for inference.

        Parameters
        ----------
        model_name : str
            The model ID to be resolved by `litellm`.
        task : TaskMetadata | str
            The task metadata object or name of an already created task.
        custom_prompt_prefix : str, optional
            A custom prompt prefix to supply to the model before the encoded
            row data, by default None.
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
            custom_prompt_prefix=custom_prompt_prefix,
            encode_row=encode_row,
            threshold=threshold,
            correct_order_bias=correct_order_bias,
            seed=seed,
            **inference_kwargs,
        )
        self.deployment_name = self._model_to_azure_deployment_name.get(
            model_name, model_name
        )

        # Initialize total cost of API calls
        self._total_cost = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._num_api_calls = 0
        self._total_response_time = 0.0

        # Set maximum requests per minute
        self.max_api_rpm = max_api_rpm
        if "MAX_API_RPM" in os.environ:
            self.max_api_rpm = int(os.getenv("MAX_API_RPM"))
            logging.info(
                f"MAX_API_RPM environment variable is set. "
                f"Overriding previous value of {max_api_rpm} with {self.max_api_rpm}."
            )
        else:
            self.max_api_rpm = max(
                self.max_api_rpm, self._model_to_max_rpm.get(model_name, 0)
            )
        # Set maximum tokens per minute
        self.max_api_tpm = max_api_tpm
        if "MAX_API_TPM" in os.environ:
            self.max_api_tpm = int(os.getenv("MAX_API_TPM"))
            logging.warning(
                f"MAX_API_TPM environment variable is set. "
                f"Overriding previous value of {max_api_tpm} with {self.max_api_tpm}."
            )
        else:
            self.max_api_tpm = max(
                self.max_api_tpm, self._model_to_max_tpm.get(model_name, 0)
            )

        # Check extra dependencies
        assert self.check_webAPI_deps(), "Web API dependencies are not installed."

        # Check OpenAI API key was passed
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
        
        # Set API type
        self.api_type = "completion"

        # litellm completion does not seem to provide reasoning with opt-in summary -> switch to responses API
        if (
            get_model_developer(self.model_name) == "OpenAI"
            and self.inference_kwargs.get("enable_thinking", False)
            and task.question.use_generated_text
        ):
            # log-probs not available via responses API, but then reasoning can only be a str!
            logging.debug(
                f"Using responses API for OpenAI reasoning model {self.model_name} with thinking enabled and text-based extraction."
            )

            self.api_type = "responses"

        # Get supported parameters
        from litellm import get_supported_openai_params

        supported_params = get_supported_openai_params(model=self.deployment_name)
        if self.api_type == "responses": 
            from litellm import OpenAIResponsesAPIConfig
            config = OpenAIResponsesAPIConfig()
            # merge lists of suppprted parameters (parameters fro the completion API should get mapped internally by the response API)
            supported_params = list(set(supported_params) | set(config.get_supported_openai_params(model=self.deployment_name)))

        if supported_params is None:
            raise RuntimeError(
                f"Failed to get supported parameters for model '{self.deployment_name}'."
            )
        self.supported_params = set(supported_params)

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
                "Please install extra API dependencies with "
                "`pip install 'folktexts[apis]'` "
                "to use the WebAPILLMClassifier."
            )
            return False
        return True

    def _query_webapi_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> list[dict]:
        """Query the web API with a batch of prompts and returns the json response.

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
        responses_batch : list[dict]
            The returned JSON responses for each prompt in the batch.
        """

        # Adapt number of forward passes
        # > Single token answers should require only one forward pass
        if question.num_forward_passes == 1:
            num_forward_passes = 1

        # NOTE: Models often generate "0." instead of directly outputting the fractional part
        # > Therefore: for multi-token answers, extra forward passes may be required
        else:
            # Add extra tokens for textual prefix, e.g., "The probability is: ..."
            num_forward_passes = question.num_forward_passes + 2

        if question.use_generated_text:
            api_call_params = dict(
                temperature=1,
                max_completion_tokens=self.inference_kwargs["max_new_tokens"],
                stream=False,
                seed=self.seed,
            )
        else:
            api_call_params = dict(
                temperature=1,
                max_completion_tokens=max(
                    num_forward_passes, self.inference_kwargs.get("max_new_tokens", 0)
                ),
                stream=False,
                seed=self.seed,
                logprobs=True,
                top_logprobs=20,
            )

        if self.model_name.startswith("claude"):
            api_call_params.pop("seed")
            logging.debug(
                "Removed 'seed' from API call parameters for Claude model, as it is not supported."
            )

        # Set extra arguments for reasoning-augmented models if thinking is enabled
        enable_thinking = self.inference_kwargs.get("enable_thinking", False)
        logging.debug("thinking is set to: " + str(enable_thinking))
        if enable_thinking:
            if self.model_name.startswith("claude"):
                budget_tokens = 4096
                logging.warning(
                    f"Thinking enabled for Claude model. Setting budget_tokens to {budget_tokens}."
                )
                # pass thinking params to Claude models
                assert budget_tokens >= 1024, "budget_tokens must be at least 1024"
                assert (
                    self.inference_kwargs["max_new_tokens"] >= budget_tokens
                ), "max_new_tokens must be greater than or equal to budget_tokens"

                api_call_params["thinking"] = {"type": "enabled", "budget_tokens": 4096}
            elif get_model_developer(self.model_name) == "OpenAI":
                reasoning_effort = "medium"
                # NOTE: reasoning_effort accepts a string value ("none", "minimal", "low", "medium", "high", "xhigh"—"xhigh" (https://docs.litellm.ai/docs/providers/openai)
                logging.warning(
                    f"Thinking enabled for OpenAI model. Setting budget_tokens to {reasoning_effort}."
                )
                if self.api_type == "responses":
                    # summary only available via responses API, but that does not support logprobs
                    # -> only use if extracting answer from generated text
                    api_call_params["reasoning"] = {
                        "effort": reasoning_effort,
                        "summary": "detailed",
                    }
                else:
                    api_call_params["reasoning_effort"] = reasoning_effort
        if set(api_call_params.keys()) - self.supported_params:
            raise RuntimeError(
                f"Unsupported API parameters for model '{self.deployment_name}': "
                f"{set(api_call_params.keys()) - self.supported_params}"
            )

        # Get system prompt depending on Q&A type
        if isinstance(question, DirectNumericQA):
            system_prompt = "Your response must start with a number representing the estimated probability."
            # system_prompt = (
            #     "You are a highly specialized assistant that always responds with a single number. "
            #     "For every input, you must analyze the request and respond with only the relevant single number, "
            #     "without any additional text, explanation, or symbols."
            # )
        elif isinstance(question, MultipleChoiceQA):
            system_prompt = "Please respond with a single letter."
        else:
            raise ValueError(f"Unknown question type '{type(question)}'.")

        # Query model for each prompt in the batch
        requests_data = [
            {
                "model": self.deployment_name,
                "api_base": (
                    os.environ["AZURE_AI_API_BASE"]
                    if self.deployment_name.startswith("azure_ai")
                    else os.environ["AZURE_API_BASE"]
                ),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **api_call_params,
            }
            for prompt in prompts_batch
        ]

        logging.debug(f"API call parameters: {api_call_params}")
        logging.debug(f"First request data: {requests_data[0]}")    

        if self.api_type == "responses":
            for p in range(len(prompts_batch)):
                requests_data[p]["input"] = requests_data[p].pop("messages")
        responses_batch = self.client.make_requests_with_retries(
            requests_data, max_retries=10, sanitize=False
        )

        return responses_batch
    
    def _decode_risk_estimate_from_api_response(
        self,
        response: dict,
        question: MultipleChoiceQA | DirectNumericQA,
    ) -> float:
        """Decode model output from API response to get risk estimate.

        Parameters
        ----------
        response : dict
            The response from the API call.
        question : MultipleChoiceQA | DirectNumericQA
            The question (`QAInterface`) object to use for querying the model.

        Returns
        -------
        risk_estimate : float
            The risk estimate for the API query.
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
                
                if self.inference_kwargs.get("enable_thinking", False) and len(reasoning_content) == 0:
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
                        if hasattr(item, 'summary') and item.summary:
                            for summary in item.summary:
                                reasoning_content += summary.text + "\n"

                response_message = (
                    "\n".join(output_texts) if len(output_texts) > 0 else ""
                    )
                if self.inference_kwargs.get("enable_thinking", False) and len(reasoning_content) == 0:
                     logging.debug("Reasoning enabled, but no summary returned.")
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
            return (
                risk_estimate,
                {"reasoning" : reasoning_content, "response" : response_message},
            )
        
        else:

            assert self.api_type != "responses", "Logprobs not available via responses API."
            # Get response message 
            choice = response.choices[0]
            response_message = choice.message.content
            logging.debug(f"Received response_message: {response_message}")

            # Get top token choices for each forward pass
            token_choices_all_passes = choice.logprobs.content
            # print(token_choices_all_passes)

            # Construct dictionary of token to linear token probability for each forward pass
            token_probs_all_passes = [
                {
                    token_metadata.token: np.exp(token_metadata.logprob)
                    for token_metadata in top_token_logprobs.top_logprobs
                }
                for top_token_logprobs in token_choices_all_passes
            ]

            # Decode model output into risk estimates
            # 1. Construct vocabulary dict for this response
            vocab_tokens = {
                tok for forward_pass in token_probs_all_passes for tok in forward_pass
            }

            token_to_id = {tok: i for i, tok in enumerate(vocab_tokens)}
            id_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}

            # 2. Parse `token_probs_all_passes` into an array of shape (num_passes, vocab_size)
            token_probs_array = np.array(
                [
                    [
                        forward_pass.get(id_to_token[i], 0)
                        for i in range(len(vocab_tokens))
                    ]
                    for forward_pass in token_probs_all_passes
                ]
            )
            # NOTE: token_probs.shape = (num_passes, vocab_size)

            # Get risk estimate
            risk_estimate = question.get_answer_from_model_output(
                token_probs_array,
                tokenizer_vocab=token_to_id,
            )

            # Sanity check numeric answers based on global model response:
            if isinstance(question, DirectNumericQA):
                try:
                    numeric_response = re.match(
                        r"[-+]?\d*\.\d+|\d+", response_message
                    ).group()
                    risk_estimate_full_text = float(numeric_response)

                    if not np.isclose(
                        risk_estimate, risk_estimate_full_text, atol=1e-2
                    ):
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

            return risk_estimate, token_probs_array

    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> np.ndarray:
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
        risk_estimates : np.ndarray
            The risk estimates for each prompt in the batch.

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
        risk_estimates_batch = []
        outputs_batch = []
        for i, response in enumerate(api_responses_batch):
            if response:
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
                        logging.debug(
                            f"Response {i+1}: {message_content[:100]}..."
                        )  # Print first 100 chars
                    else: 
                        logging.debug(
                            f"Response {i+1} is None."
                        )
                    risk_est, out = self._decode_risk_estimate_from_api_response(
                        response, question
                    )
                    risk_estimates_batch.append(risk_est)
                    outputs_batch.append(
                        out
                    )  # if not question.use_generated_text else out.replace(";", ""))
                except (AttributeError, IndexError, TypeError) as e:
                    logging.error(
                        f"Response {i+1}: Could not parse response content. Error: {e}"
                    )
                    logging.error(f"Raw response: {response}")
                    logging.error("Adding NaN value.")
                    risk_estimates_batch.append(np.nan)
                    outputs_batch.append(None)
            else:
                logging.error(f"Response {i+1}: Request failed. Adding NaN value. ")
                risk_estimates_batch.append(np.nan)
                outputs_batch.append(None)

        self.track_stats()
        return risk_estimates_batch, outputs_batch

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

    def track_stats(self):
        # get all tracker attributes with defaults
        total_cost = getattr(self.client.tracker, "total_cost", 0)
        total_prompt_tokens = getattr(self.client.tracker, "total_prompt_tokens", 0)
        total_completion_tokens = getattr(
            self.client.tracker, "total_completion_tokens", 0
        )
        num_api_calls = getattr(self.client.tracker, "num_api_calls", 0)
        mean_response_time = getattr(self.client.tracker, "mean_response_time", None)

        logging.info(f"Total cost: ${total_cost:.4f}")
        logging.info(f"Total prompt tokens: {total_prompt_tokens}")
        logging.info(f"Total completion tokens: {total_completion_tokens}")
        logging.info(f"Number of successful API calls: {num_api_calls}")

        if mean_response_time is not None and isinstance(
            mean_response_time, (int, float)
        ):
            logging.info(f"Mean response time: {mean_response_time:.2f}s")
        else:
            logging.info("Mean response time: Not available")

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
            elif (
                isinstance(completion_response, dict) and "cost" in completion_response
            ):
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
