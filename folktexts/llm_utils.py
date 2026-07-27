"""Common functions to use with transformer LLMs."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA
# Will warn if the sum of digit probabilities is below this threshold
PROB_WARN_THR = 0.5


def reasoning_to_enable_thinking(reasoning: str | None) -> bool | None:
    """Translate the ``reasoning`` config knob into the chat template's
    ``enable_thinking`` switch.

    - ``None`` (reasoning unset)              -> ``None``  (omit the kwarg; template default)
    - ``"0"`` / ``0`` (thinking disabled)     -> ``False``
    - any other value (effort/budget)         -> ``True``

    ``reasoning`` also encodes *how much* effort/budget for other model families
    (OpenAI ``reasoning_effort``, Claude ``budget_tokens``); this collapse to a
    tri-state boolean is only for the local chat-template ``enable_thinking``
    kwarg. Note it does **not** decide whether a template is applied — that
    depends on the tokenizer having a ``chat_template`` (base vs instruction).

    The ``"0"`` sentinel is compared as a string because the CLI's
    int-or-float-or-str parser may hand us the integer ``0`` for ``--reasoning=0``.
    """
    if reasoning is None:
        return None
    return str(reasoning) != "0"


THINK_START_MARKER = "<think>"
THINK_END_MARKER = "</think>"

# How to locate the end-of-thinking boundary when stripping a `<think>...</think>`
# block from a generation:
#   - "token_id": split on the `</think>` *token* (robust — a literal `</think>`
#     appearing inside the reasoning text can't fool it; needs the token ids +
#     tokenizer, and yields the reasoning-token count for free).
#   - "string": split on the literal `</think>` marker in the decoded text (works
#     with only the text, e.g. a backend that surfaces no token ids).
# Both split on the LAST occurrence of the boundary. This is the single toggle
# shared by every local backend; token-id mode gracefully degrades to string mode
# when token ids/tokenizer are unavailable.
DEFAULT_THINKING_SPLIT_MODE = "token_id"


def _postprocess_generated_text(
    *,
    enable_thinking: bool | None,
    i: int,
    n: int,
    split_by: str = DEFAULT_THINKING_SPLIT_MODE,
    text: str | None = None,
    token_ids: list[int] | None = None,
    tokenizer: AutoTokenizer | None = None,
    thinking_end_token_id: int | None = None,
) -> dict:
    """Split a generation into response / reasoning for downstream extraction.

    In thinking mode the model emits a `<think> ... </think>` block before the
    answer. We keep only what follows the LAST `</think>` for extraction and
    return the reasoning (everything before it) separately for logging/metadata.
    Outside thinking mode the whole generation is the response.

    Parameters
    ----------
    enable_thinking : bool | None
        Whether thinking mode is enabled. Only ``True`` triggers a split.
    i, n : int
        Position / size of this generation in the batch (for debug logging).
    split_by : {"token_id", "string"}
        How to locate the `</think>` boundary (see ``DEFAULT_THINKING_SPLIT_MODE``).
        ``"token_id"`` needs ``token_ids`` + ``tokenizer`` and degrades to
        ``"string"`` if either is missing.
    text : str, optional
        Decoded generation (new tokens only). Required for ``split_by="string"``;
        also the no-marker fallback text in either mode.
    token_ids : list[int], optional
        Generated token ids (prompt excluded). Required for ``split_by="token_id"``.
    tokenizer : AutoTokenizer, optional
        Needed to decode in ``token_id`` mode, to resolve the marker id, and to
        count reasoning tokens in ``string`` mode.
    thinking_end_token_id : int, optional
        The `</think>` token id; resolved from ``tokenizer`` when omitted.

    Returns
    -------
    dict with keys ``response`` (str, for extraction), ``reasoning`` (str) and
    ``reasoning_tokens`` (int | None — ``None`` only when it cannot be counted).
    """
    if split_by not in ("token_id", "string"):
        raise ValueError(f"Unknown split_by={split_by!r}; expected 'token_id' or 'string'.")

    # Full decoded generation (marker not yet stripped) — for logging and the
    # non-thinking / no-marker fallback paths.
    if text is not None:
        full_text = text
    elif token_ids is not None and tokenizer is not None:
        full_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    else:
        raise ValueError("Provide `text` (string mode) or `token_ids`+`tokenizer` (token_id mode).")

    def _count_tokens(s: str) -> int | None:
        if not s:
            return 0
        if tokenizer is None:
            return None
        return len(tokenizer.encode(s, add_special_tokens=False))

    def _finish(response: str, reasoning: str, reasoning_tokens: int | None) -> dict:
        # Drop the wrapping `<think>`/`</think>` markers from the stored reasoning
        # so the `reasoning` column holds only the thinking content. (The token-id
        # slice includes the trailing `</think>` token, and both split paths keep
        # the leading `<think>`.) `reasoning_tokens` still counts the raw slice.
        reasoning = reasoning.strip()
        if reasoning.startswith(THINK_START_MARKER):
            reasoning = reasoning[len(THINK_START_MARKER) :]
        if reasoning.endswith(THINK_END_MARKER):
            reasoning = reasoning[: -len(THINK_END_MARKER)]
        reasoning = reasoning.strip()
        logging.debug(f"=== Generated output {i + 1}/{n} ===")
        logging.debug(f"Thinking content ({len(reasoning)} chars) [IGNORED for extraction]:")
        logging.debug(f"{reasoning[:500]}..." if len(reasoning) > 500 else reasoning)
        logging.debug(f"Response content ({len(response)} chars) [USED for extraction]:")
        logging.debug(response)
        if not response:
            logging.warning(
                "Response content after </think> is empty. "
                "Model may not have generated a proper response after reasoning. "
                "Probability extraction will likely fail."
            )
        return {"response": response, "reasoning": reasoning, "reasoning_tokens": reasoning_tokens}

    def _no_thinking() -> dict:
        logging.debug(f"=== Generated output {i + 1}/{n} ===")
        logging.debug(f"Content ({len(full_text)} chars):\n{full_text[:500]}...")
        return {"response": full_text.strip(), "reasoning": "", "reasoning_tokens": 0}

    # A `</think>` block may be present even when thinking was NOT requested — a
    # model can emit one regardless (e.g. `reasoning='0'` that it doesn't fully
    # honor). Strip it whenever it is actually present so reasoning never leaks
    # into the extracted response; `enable_thinking` only drives the
    # expected-but-missing warning below.

    # Token-id split (preferred): operate on the token stream, last `</think>`.
    if split_by == "token_id" and token_ids is not None and tokenizer is not None:
        if thinking_end_token_id is None:
            thinking_end_token_id = get_thinking_end_token_id(tokenizer=tokenizer)
        if thinking_end_token_id is not None and thinking_end_token_id in token_ids:
            # Index just past the LAST `</think>` token (search from the end).
            index = len(token_ids) - token_ids[::-1].index(thinking_end_token_id)
            reasoning = tokenizer.decode(token_ids[:index], skip_special_tokens=True).strip()
            response = tokenizer.decode(token_ids[index:], skip_special_tokens=True).strip()
            # Reasoning-token count is free from the slice (incl. the marker token).
            return _finish(response, reasoning, index)
        # No marker in the token stream → fall through to the string/fallback path.

    # String split (or token-id fallback): split the decoded text on the last marker.
    if THINK_END_MARKER in full_text:
        reasoning, _sep, response = full_text.rpartition(THINK_END_MARKER)
        reasoning, response = reasoning.strip(), response.strip()
        return _finish(response, reasoning, _count_tokens(reasoning))

    # No `</think>` present → the whole generation is the response.
    if enable_thinking is True:
        logging.warning(
            f"</think> marker not found in output (thinking mode was enabled). "
            f"Using full generated text ({len(full_text)} chars)."
        )
    return _no_thinking()


def query_model_batch(
    text_inputs: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_size: int,
) -> np.ndarray:
    """Queries the model with a batch of text inputs.

    Parameters
    ----------
    text_inputs : list[str]
        The inputs to the model as a list of strings.
    model : AutoModelForCausalLM
        The model to query.
    tokenizer : AutoTokenizer
        The tokenizer used to encode the text inputs.
    context_size : int
        The maximum context size to consider for each input (in tokens).

    Returns
    -------
    last_token_probs : np.ndarray
        Model's last token *linear* probabilities for each input as an
        np.array of shape (batch_size, vocab_size).
    """
    model_device = next(model.parameters()).device

    # Batched tokenization. `truncation_side="left"` reproduces the previous
    # per-row `[-context_size:]` semantics (keep the tail, drop the head);
    # `padding_side="right"` is required by the last-token read below —
    # `idx = attention_mask.sum(-1) - 1` points at the last real token only
    # when pads sit AFTER the prompt (single forward pass, not `.generate()`,
    # which uses left-padding in `generate_text_batch`). Restore both
    # attributes even if the tokenizer call raises.
    old_pad_side = tokenizer.padding_side
    old_trunc_side = tokenizer.truncation_side
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    try:
        tokenized = tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=context_size,
            add_special_tokens=True,
        )
    finally:
        tokenizer.padding_side = old_pad_side
        tokenizer.truncation_side = old_trunc_side

    # Compute the last-real-token index on CPU before moving tensors to the
    # model device, so the downstream `logits[torch.arange(...), idx]` gather
    # uses plain Python ints and doesn't mix CPU/CUDA index tensors.
    idx_last_token = (tokenized.attention_mask.sum(dim=1) - 1).tolist()
    tensor_inputs = tokenized.input_ids.to(model_device)
    attention_mask = tokenized.attention_mask.to(model_device)

    # Query: run one forward pass, i.e., generate the next token
    with torch.no_grad():
        logits = model(input_ids=tensor_inputs, attention_mask=attention_mask).logits

    # Probabilities corresponding to the last token after the prompt
    last_token_logits = logits[torch.arange(len(idx_last_token)), idx_last_token]
    last_token_probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    return last_token_probs.to(dtype=torch.float16).cpu().numpy()


def query_model_batch_multiple_passes(
    text_inputs: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_size: int,
    n_passes: int,
    digits_only: bool = False,
) -> np.ndarray:
    """Queries an LM for multiple forward passes.

    Greedy token search over multiple forward passes: Each forward pass takes
    the highest likelihood token from the previous pass.

    NOTE: could use model.generate in the future!

    Parameters
    ----------
    text_inputs : list[str]
        The batch inputs to the model as a list of strings.
    model : AutoModelForCausalLM
        The model to query.
    tokenizer : AutoTokenizer
        The tokenizer used to encode the text inputs.
    context_size : int
        The maximum context size to consider for each input (in tokens).
    n_passes : int, optional
        The number of forward passes to run.
    digits_only : bool, optional
        Whether to only sample for digit tokens.

    Returns
    -------
    last_token_probs : np.array
        Last token *linear* probabilities for each forward pass, for each text
        in the input batch. The output has shape (batch_size, n_passes, vocab_size).
    """
    # Mask is sized to the model's logits dim, not the tokenizer's vocab dict:
    # neither `len(tokenizer.vocab)` nor `tokenizer.vocab_size` is reliable
    # (Gemma-3 has `len(vocab) == vocab_size + 1`; Llama-3.2 has
    # `len(vocab) == vocab_size + 256`). Only `model.config.vocab_size` matches
    # the actual logits axis we're masking. Multimodal Gemma-3 puts vocab_size
    # under `config.text_config` instead of the top-level config.
    vocab_dim = getattr(model.config, "vocab_size", None)
    if vocab_dim is None:
        vocab_dim = getattr(getattr(model.config, "text_config", None), "vocab_size", None)
    if vocab_dim is None:
        raise AttributeError(
            f"Could not resolve vocab_size from {type(model.config).__name__} (checked top-level and text_config)."
        )
    allowed_tokens_filter = np.ones(vocab_dim, dtype=bool)
    if digits_only:
        allowed_token_ids = np.array(
            [tok_id for token, tok_id in tokenizer.vocab.items() if token.isdecimal() and tok_id < vocab_dim]
        )

        allowed_tokens_filter = np.zeros(vocab_dim, dtype=bool)
        allowed_tokens_filter[allowed_token_ids] = True

    # Current text batch
    current_batch = text_inputs

    # For each forward pass, add one token to each text in the batch
    last_token_probs = []

    for iter in range(n_passes):
        # Query the model with the current batch
        current_probs = query_model_batch(current_batch, model, tokenizer, context_size)

        # Filter out probabilities for tokens that are not allowed
        current_probs[:, ~allowed_tokens_filter] = 0

        # Sanity check digit probabilities
        if iter == 0 and digits_only:
            total_digit_probs = np.sum(current_probs, axis=-1)
            if any(probs < PROB_WARN_THR for probs in total_digit_probs):
                logging.error(f"Digit probabilities are too low: {total_digit_probs}")

        # Add the highest likelihood token to each text in the batch
        next_tokens = [tokenizer.decode([np.argmax(probs)]) for probs in current_probs]
        current_batch = [text + next_token for text, next_token in zip(current_batch, next_tokens)]

        # Store the probabilities of the last token for each text in the batch
        last_token_probs.append(current_probs)

    # Cast output to np.array with correct shape
    last_token_probs_array: np.ndarray = np.array(last_token_probs)
    last_token_probs_array = np.moveaxis(last_token_probs_array, 0, 1)
    assert last_token_probs_array.shape == (len(text_inputs), n_passes, vocab_dim)
    return last_token_probs_array


def decode_topk_logprobs_to_risk_estimate(
    per_pass_topk: list[dict[int, float]],
    *,
    tokenizer_vocab: dict[str, int],
    vocab_dim: int,
    question: "MultipleChoiceQA | DirectNumericQA",
) -> float:
    """Convert top-K log-probabilities into a single risk-estimate float.

    Parameters
    ----------
    per_pass_topk : list[dict[int, float]]
        One dict per generated token position, mapping token_id -> log-prob. The
        token_ids must match the values in `tokenizer_vocab`. Tokens absent from
        the top-K are assumed to have probability ~0.
    tokenizer_vocab : dict[str, int]
        Token string -> token_id map used by the QA decoder for prefix-variant
        lookup (MultipleChoiceQA) or digit/decimal lookup (DirectNumericQA).
    vocab_dim : int
        Size of the linear-probability array's vocab axis. For local backends
        this is `model.config.vocab_size` (the logits axis); for the synthetic
        WebAPI path it is the size of the synthesised vocab.
    question : MultipleChoiceQA | DirectNumericQA
        The QA interface used to interpret the probabilities.

    Returns
    -------
    risk_estimate : float
        Risk score in [0, 1] from `question.get_answer_from_model_output`.

    Notes
    -----
    Both the WebAPI backend (top_logprobs=20 from OpenAI-style responses) and
    the vLLM backend (top-K logprobs from `SamplingParams(logprobs=K)`) call
    this helper. The transformers backend reads the full softmax directly and
    bypasses this path; see `query_model_batch_multiple_passes`.
    """
    n_passes = len(per_pass_topk)
    probs = np.zeros((n_passes, vocab_dim), dtype=np.float64)
    for i, pass_dict in enumerate(per_pass_topk):
        for tok_id, logprob in pass_dict.items():
            if 0 <= tok_id < vocab_dim:
                probs[i, tok_id] = float(np.exp(logprob))

    # Drop tokenizer-vocab entries that point past the array. MultipleChoiceQA's
    # decoder does an unchecked `last_token_probs[choice_token_id]` lookup; on
    # tokenizers where added tokens sit beyond `model.config.vocab_size`
    # (Llama-3.2, Gemma-3) those ids would IndexError. DirectNumericQA already
    # filters the same way internally — this keeps both modes consistent.
    in_range_vocab = {tok: tok_id for tok, tok_id in tokenizer_vocab.items() if 0 <= tok_id < vocab_dim}

    return question.get_answer_from_model_output(
        probs,
        tokenizer_vocab=in_range_vocab,
    )


# generate_text_batch
def generate_text_batch(
    text_inputs: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    context_size: int = None,
    temperature: float = 0.0,
    seed: int | None = None,
) -> list[dict]:
    """Generate text completions for a batch of prompts.

    The chat template (if any) is applied upstream by the encode_row function;
    this tokenizes ``text_inputs`` as-is and generates.

    Uses the model's generate() method for autoregressive text generation,
    particularly suitable for reasoning models and chain-of-thought Q&A where
    the model needs to producefree-form text before outputting a probability
    estimate. Generation is greedy when temperature <= 0 (the default);
    otherwise it sampled at the given temperature and can be seeded via ``seed``
    for reproducibility.

    Parameters
    ----------
    text_inputs : list[str]
        The input prompts as a list of strings.
    model : AutoModelForCausalLM
        The model to use for generation.
    tokenizer : AutoTokenizer
        The tokenizer used to encode/decode text.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default 1024.
    context_size : int, optional
        The maximum context size for input tokens. If None, no truncation
        is applied to inputs.
    temperature : float, optional
        Sampling temperature. Values <= 0 use greedy decoding; values > 0
        enable sampling at the given temperature. Defaults to 0.0.
    seed : int | None, optional
        Random seed to set immediately before generation when sampling is
        enabled. Ignored for greedy generation.

    Returns
    -------
    generations : list[dict]
        One raw generation per prompt with keys ``text`` (the full decoded
        generation) and ``token_ids`` (the new token ids). Only newly generated
        tokens are considered (the input prompt is excluded). The `<think>`
        split is left to the caller via `_postprocess_generated_text`.
    """
    model_device = next(model.parameters()).device

    # Save original padding/truncation sides; force left for generation.
    # Decoder-only models require left-padding for correct generation, and
    # left-truncation keeps the prompt tail (the question at the end) when a
    # prompt exceeds context_size.
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # The chat template (if any) is already applied upstream by the encode_row
    # function, so the prompts are tokenized as-is.

    try:
        # tokenize inputs (already chat-formatted, or raw for base models)
        tokenized = tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True if context_size else False,
            max_length=context_size,
        )

        tensor_inputs = tokenized.input_ids.to(model_device)
        attention_mask = tokenized.attention_mask.to(model_device)
        input_seq_length = tensor_inputs.shape[1]

        do_sample = temperature is not None and temperature > 0.0
        generate_kwargs = dict(
            input_ids=tensor_inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )

        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            if do_sample and seed is not None:
                torch.manual_seed(seed)
            outputs = model.generate(**generate_kwargs)

        # Return the raw per-row generations (full decoded text + new token ids).
        # The `<think>...</think>` split is done by the caller at its call site via
        # `_postprocess_generated_text`, mirroring the vLLM backend so both
        # classifiers share one post-processing path (and each holds the full
        # generated text locally for logging).
        generations: list[dict] = []
        for output in outputs:
            generated_tokens = output[input_seq_length:].tolist()
            generations.append(
                {
                    "text": tokenizer.decode(generated_tokens, skip_special_tokens=True),
                    "token_ids": generated_tokens,
                }
            )

        return generations

    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side


def add_pad_token(tokenizer):
    """Add a pad token to the model and tokenizer if it doesn't already exist.

    Here we're using the end-of-sentence token as the pad token. Both the model
    weights and tokenizer vocabulary are untouched.

    Another possible way would be to add a new token `[PAD]` to the tokenizer
    and update the tokenizer vocabulary and model weight embeddings accordingly.
    The embedding for the new pad token would be the average of all other
    embeddings.
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})


def is_bf16_compatible() -> bool:
    """Checks if the current environment is bfloat16 compatible."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_model_tokenizer(
    model_name_or_path: str | Path, padding_side=None, **kwargs
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer from the given local path (or using the model name).

    Parameters
    ----------
    model_name_or_path : str | Path
        Model name or local path to the model folder.
    padding_side : {"left", "right"}, optional
        Initial tokenizer padding side. Normally unnecessary: the inference
        paths each set the side they require and restore it per call
        (`query_model_batch_multiple_passes` forces "right" for the last-token
        read; `generate_text_batch` forces "left" for decoder-only generation),
        so the load-time value is overridden either way. Left as an escape hatch
        for callers that tokenize directly. If None, the tokenizer's own default
        is kept.
    kwargs : dict
        Additional keyword arguments to pass to the model `from_pretrained` call.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        The loaded model and tokenizer, respectively.
    """
    logging.info(f"Loading model '{model_name_or_path}'")

    # Load tokenizer from disk
    if padding_side is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)

    # Set default keyword arguments for loading the pretrained model
    model_kwargs = dict(
        dtype=torch.bfloat16 if is_bf16_compatible() else torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model_kwargs.update(kwargs)

    # Load model from disk
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    # if not model.config.is_encoder_decoder:
    #     tokenizer.padding_side = 'left'
    #     logging.debug(f"Set tokenizer.padding_side to 'left' for model {model_name_or_path}"
    #                   " since it's not an encoder-decoder model.")

    # Add pad token to the tokenizer if it doesn't already exist
    add_pad_token(tokenizer)

    # Move model to the correct device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    logging.info(f"Moving model to device: {device}")
    if model.device.type != device:
        model.to(device)

    return model, tokenizer


def load_vllm_model(
    model_name_or_path: str | Path,
    *,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    seed: int = 42,
    max_logprobs: int = 50,
    **kwargs,
):
    """Load a vLLM `LLM` engine and its tokenizer.

    Mirrors `load_model_tokenizer` for the vLLM backend. vLLM allocates the KV
    cache statically at startup based on `gpu_memory_utilization` and
    `max_model_len`; tune these per-GPU. `vllm` is an optional install — if it
    is not importable, this function raises a pointed error.

    Parameters
    ----------
    model_name_or_path : str | Path
        Model name or local path to the model folder. Pre-cached snapshots
        under `/fast/groups/sf/huggingface-models/` work without download.
    dtype : str, optional
        Compute dtype: ``"auto"`` (default; vLLM picks bf16/fp16 from the
        config), ``"bfloat16"``, ``"float16"``, or ``"float32"``.
    gpu_memory_utilization : float, optional
        Fraction of GPU VRAM vLLM may use for weights + KV cache. Default 0.85
        (vLLM's own default is 0.9, which is aggressive on shared cluster
        nodes). vLLM fails fast at startup if this isn't enough — bump down
        if you hit OOM at LLM().
    max_model_len : int, optional
        Maximum number of tokens (input + output) per request. If ``None``,
        vLLM reads it from the model config — which on some Llama checkpoints
        is 131072 and will allocate enormous KV cache. Pass an explicit value
        sized as ``context_size + max_new_tokens + buffer`` for the workload.
    tensor_parallel_size : int, optional
        Number of GPUs to shard the model across; default 1. Set higher when
        the cluster job grants multiple GPUs and the model fits with
        tensor-parallel sharding.
    trust_remote_code : bool, optional
        Forwarded to vLLM (mirrors `load_model_tokenizer`).
    seed : int, optional
        Random seed for vLLM. Doesn't affect greedy (`temperature=0`) decoding
        — used by multiple-choice / numeric QA — but governs reproducibility of
        sampled paths such as chain-of-thought (`temperature=1` by default).
    max_logprobs : int, optional
        Engine-level cap on top-K logprobs SamplingParams may request.
        Default 50 — must be ≥ ``VLLMClassifier._TOPK_LOGPROBS`` or the engine
        rejects the request at predict time (`VLLMValidationError: Requested
        sample logprobs of K, which is greater than max allowed`).
    **kwargs
        Additional keyword arguments forwarded verbatim to ``vllm.LLM(...)``.

    Returns
    -------
    tuple[vllm.LLM, AutoTokenizer]
        Loaded engine and its tokenizer. The tokenizer has had `add_pad_token`
        applied so it matches the transformers path's tokenizer state.
    """
    try:
        from vllm import LLM
    except ImportError as exc:  # pragma: no cover - exercised in user-facing CLI
        raise ImportError(
            "vLLM is not installed. Install the optional extra with "
            "`pip install 'folktexts[vllm]'`, or run with "
            "`--inference-backend transformers` to use the HuggingFace path."
        ) from exc

    # vLLM is extremely chatty during model loading; quieten it unless the
    # caller has explicitly opted into verbose logs.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    # `processed_logprobs` returns top-K logprobs computed AFTER `allowed_token_ids`
    # masking. The default `raw_logprobs` would return top-K from the unmasked
    # distribution — which on `DirectNumericQA` causes the decoder to see non-digit
    # tokens (e.g., '.', '\n') as high-probability "numeric tokens" and pick them
    # over the only-allowed digit, collapsing Llama-3 base numeric output to 0.5
    # (answer text "5." → regex "5" → 0.5). MC has no `allowed_token_ids` so this
    # defaults to the raw distribution either way; numeric is the only mode this
    # affects.
    kwargs.setdefault("logprobs_mode", "processed_logprobs")

    # `VLLM_ENFORCE_EAGER=1` disables torch.compile + CUDA-graph capture. vLLM
    # itself doesn't read this env var, so honor it here: it sidesteps the
    # inductor -> Triton -> gcc JIT build that crashes on nodes with a broken
    # toolchain / unlinkable libcuda.so.1 (see scripts/check_gcc.sh). An explicit
    # caller-supplied `enforce_eager` kwarg still wins.
    if os.environ.get("VLLM_ENFORCE_EAGER", "").lower() in ("1", "true", "yes"):
        kwargs.setdefault("enforce_eager", True)

    logging.info(f"Loading vLLM model '{model_name_or_path}'")
    llm = LLM(
        model=str(model_name_or_path),
        dtype=dtype,  # type: ignore[arg-type]  # str accepted at runtime (auto/bfloat16/…)
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        seed=seed,
        max_logprobs=max_logprobs,
        **kwargs,
    )
    tokenizer = llm.get_tokenizer()
    add_pad_token(tokenizer)
    return llm, tokenizer


def get_model_folder_path(model_name: str, root_dir="/tmp") -> str:
    """Returns the folder where the model is saved."""
    folder_name = model_name.replace("/", "--")
    return (Path(root_dir) / folder_name).resolve().as_posix()


def get_model_size_B(model_name: str, default: int = None) -> int | float | None:
    """Get the model size from the model name, in Billions of parameters."""
    regex = re.search(r"((?P<times>\d+)[xX])?(?P<size>(\d\.)?\d+)[bB]", model_name)
    if regex:
        size = regex.group("size")
        return (float(size) if "." in size else int(size)) * int(regex.group("times") or 1)

    if default is not None:
        return default

    logging.warning(f"Could not infer model size from name '{model_name}'.")
    return default


def get_thinking_end_token_id(tokenizer: AutoTokenizer) -> int | None:
    try:
        think_id = tokenizer.encode("</think>", add_special_tokens=False)
    except Exception:
        # Tokenizer can't encode it (e.g. a minimal stub) -> no thinking token.
        return None
    if len(think_id) == 1:
        return think_id[0]
    logging.debug("Could not identify token id marking the end of thinking content.")
    return None


def get_model_developer(model_name: str):
    model_part = model_name.split("/")[-1].lower()

    model_prefix_to_developer = {
        "gpt": "OpenAI",
        "o1": "OpenAI",
        "o3": "OpenAI",
        "o4": "OpenAI",
        "yi": "01-ai",
        "deepseek": "DeepSeek",
        "claude": "Anthropic",
        "qwen": "Alibaba Cloud",
        "mistral": "Mistral AI",
        "mixtral": "Mistral AI",
        "gemma": "Google",
        "olmo": "Allen Institute for AI",
        "kimi": "Moonshot AI",
        "grok": "xAI",
    }
    # Substring matches (checked if no prefix match is found)

    model_substring_to_developer = {
        "llama": "Meta",
    }

    for prefix, developer in model_prefix_to_developer.items():
        if model_part.startswith(prefix):
            return developer

    # Check substring matches
    for substring, developer in model_substring_to_developer.items():
        if substring in model_part:
            return developer

    raise ValueError("Model name couldn't be matched. Please update the model-developer mapping in get_model_developer().")
