"""Common functions to use with transformer LLMs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompting import apply_chat_template

# Will warn if the sum of digit probabilities is below this threshold
PROB_WARN_THR = 0.5


def query_model_batch(
    text_inputs: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_size: int,
) -> np.array:
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
    last_token_probs : np.array
        Model's last token *linear* probabilities for each input as an
        np.array of shape (batch_size, vocab_size).
    """
    model_device = next(model.parameters()).device

    # Tokenize
    token_inputs = [
        tokenizer.encode(text, return_tensors="pt").flatten()[-context_size:]
        for text in text_inputs
    ]
    idx_last_token = [tok_seq.shape[0] - 1 for tok_seq in token_inputs]

    # Pad
    tensor_inputs = torch.nn.utils.rnn.pad_sequence(
        token_inputs,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(model_device)

    # Mask padded context
    attention_mask = tensor_inputs.ne(tokenizer.pad_token_id)

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
) -> np.array:
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
    # If `digits_only`, get token IDs for digit tokens
    allowed_tokens_filter = np.ones(len(tokenizer.vocab), dtype=bool)
    vocab_mismatch = False
    if digits_only:
        allowed_token_ids = np.array([
            tok_id
            for token, tok_id in tokenizer.vocab.items() if token.isdecimal()
        ])

        allowed_tokens_filter = np.zeros(len(tokenizer.vocab), dtype=bool)
        allowed_tokens_filter[allowed_token_ids] = True

    # Current text batch
    current_batch = text_inputs

    # For each forward pass, add one token to each text in the batch
    last_token_probs = []

    for iter in range(n_passes):
        # Query the model with the current batch
        current_probs = query_model_batch(current_batch, model, tokenizer, context_size)

        try:
            # Filter out probabilities for tokens that are not allowed
            current_probs[:, ~allowed_tokens_filter] = 0
        except IndexError:
            logging.error(
                "Size of tokenizer.vocab and model output don't match. Fix by recreating allowd_token_filter."
            )
            vocab_mismatch = True
            actual_vocab_size = current_probs.shape[1]
            allowed_tokens_filter = np.ones(actual_vocab_size, dtype=bool)
            if digits_only:
                allowed_token_ids = np.array(
                    [
                        tok_id
                        for token, tok_id in tokenizer.vocab.items()
                        if token.isdecimal()
                    ]
                )

                allowed_tokens_filter = np.zeros(current_probs.shape[1], dtype=bool)
                allowed_tokens_filter[allowed_token_ids] = True
            current_probs[:, ~allowed_tokens_filter] = 0

        # Sanity check digit probabilities
        if iter == 0 and digits_only:
            total_digit_probs = np.sum(current_probs, axis=-1)
            if any(probs < PROB_WARN_THR for probs in total_digit_probs):
                logging.error(f"Digit probabilities are too low: {total_digit_probs}")

        # Add the highest likelihood token to each text in the batch
        next_tokens = [tokenizer.decode([np.argmax(probs)]) for probs in current_probs]
        current_batch = [
            text + next_token for text, next_token in zip(current_batch, next_tokens)
        ]

        # Store the probabilities of the last token for each text in the batch
        last_token_probs.append(current_probs)

    # Cast output to np.array with correct shape
    last_token_probs_array = np.array(last_token_probs)
    last_token_probs_array = np.moveaxis(last_token_probs_array, 0, 1)
    assert last_token_probs_array.shape == (
        len(text_inputs),
        n_passes,
        len(tokenizer.vocab) if not vocab_mismatch else actual_vocab_size,
    )
    return last_token_probs_array


def query_model_text_batch(
    text_inputs: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,  # = 2048,
    context_size: int = None,
    enable_thinking: bool = None,
    thinking_end_token_id: int = None,
    system_prompt: str = None,
) -> list[str]:
    """Generate text completions for a batch of prompts.

    Uses the model's generate() method for autoregressive text generation,
    particularly suitable for reasoning-based Q&A. Text is sampled
    using the model's default generation parameters (temperature, top_p, etc.).

    Parameters
    ----------
    text_inputs : list[str]
        The input prompts as a list of strings.
    model : AutoModelForCausalLM
        The model to use for generation.
    tokenizer : AutoTokenizer
        The tokenizer used to encode/decode text.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default 2048.
    context_size : int, optional
        The maximum context size for input tokens. If None, no truncation
        is applied to inputs.
    enable_thinking : bool, optional
        Whether to enable thinking mode for models that support it (e.g., Qwen3).
        When True, uses `tokenizer.apply_chat_template` with `enable_thinking=True`.
        When False, explicitly disables thinking mode. When None (default),
        does not apply chat template formatting.

    Returns
    -------
    generated_texts : list[str]
        The generated text completions for each input prompt. Only the
        newly generated tokens are returned (not the input prompt).
    """
    model_device = next(model.parameters()).device

    # Apply chat template if enable_thinking is specified
    if enable_thinking is not None:
        processed_inputs = []

        supports_thinking = supports_enable_thinking(tokenizer=tokenizer)
        chat_kwargs = {"enable_thinking": enable_thinking} if supports_thinking else {}

        if not supports_thinking:
            logging.warning(
                "Tokenizer does not support 'enable_thinking' parameter. "
                "Falling back to standard chat template."
            )
        for text in text_inputs:
            # Format as chat messages
            formatted_text = apply_chat_template(
                tokenizer=tokenizer,
                user_prompt=text,
                tokenize=False,
                add_generation_prompt=True,
                chat_prompt=None,
                system_prompt=system_prompt,
                **chat_kwargs,
            )
            processed_inputs.append(formatted_text)

        text_inputs = processed_inputs

    # Tokenize inputs
    cutoff = -context_size if context_size is not None else None
    token_inputs = [
        tokenizer.encode(text, return_tensors="pt").flatten()[cutoff:]
        for text in text_inputs
    ]

    # Track input lengths to extract only generated tokens later
    input_lengths = [len(tokens) for tokens in token_inputs]
    if tokenizer.padding_side == "left":
        # all inputs get left-padded and end on the same padded position
        input_lengths = [max(input_lengths)] * len(token_inputs)

    # Pad sequences for batched generation
    tensor_inputs = torch.nn.utils.rnn.pad_sequence(
        token_inputs,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side,
    ).to(model_device)

    # Create attention mask
    attention_mask = tensor_inputs.ne(tokenizer.pad_token_id)

    # Generate text using model's default sampling parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tensor_inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode generated tokens (excluding input prompt)
    generated_texts = []
    for i, output in enumerate(outputs):
        # Extract only the newly generated tokens
        generated_tokens = output[input_lengths[i] :].tolist()

        # If thinking mode was enabled, separate thinking content from response content
        # The </think> token (ID 151668) marks the end of thinking content
        if enable_thinking:
            if not thinking_end_token_id:
                thinking_end_token_id = get_thinking_end_token_id(tokenizer=tokenizer)
                if thinking_end_token_id is None:
                    logging.warning(
                        "Could not identify </think> token ID. Thinking content will not be separated from response."
                    )
                # None if not found -> index(None) will throw ValueError
            try:
                # Find the </think> token from the end (in case there are multiple)
                index = len(generated_tokens) - generated_tokens[::-1].index(thinking_end_token_id)
                # Only decode content after </think>
                content_tokens = generated_tokens[index:]
                thinking_tokens = generated_tokens[:index]

                thinking_content = tokenizer.decode(thinking_tokens, skip_special_tokens=True).strip("\n")
                content = tokenizer.decode(content_tokens, skip_special_tokens=True).strip("\n")

                # Log all decoded tokens at debug level
                logging.debug(f"=== Generated output {i+1}/{len(outputs)} ===")
                logging.debug(f"Thinking content ({len(thinking_content)} chars):\n{thinking_content}")
                logging.debug(f"Response content ({len(content)} chars):\n{content}")

                generated_texts.append({"reasoning" : thinking_content, "response": content})
            except ValueError:
                # </think> token not found - decode entire output
                logging.warning("</think> token not found in output. Using full generated text.")
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                logging.debug(f"=== Generated output {i+1}/{len(outputs)} (no thinking separation) ===")
                logging.debug(f"Full content ({len(generated_text)} chars):\n{generated_text}")
                generated_texts.append({"response": generated_text})
        else:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logging.debug(f"=== Generated output {i+1}/{len(outputs)} ===")
            logging.debug(f"Content ({len(generated_text)} chars):\n{generated_text}")
            generated_texts.append({"response": generated_text})

    return generated_texts


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
    #     logging.debug(f"Set tokenizer.padding_side to 'left' for model {model_name_or_path} since it's not an encoder-decoder model.")

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


def get_model_folder_path(model_name: str, root_dir="/tmp") -> str:
    """Returns the folder where the model is saved."""
    folder_name = model_name.replace("/", "--")
    return (Path(root_dir) / folder_name).resolve().as_posix()


def get_model_size_B(model_name: str, default: int = None) -> int:
    """Get the model size from the model name, in Billions of parameters."""
    regex = re.search(r"((?P<times>\d+)[xX])?(?P<size>(\d\.)?\d+)[bB]", model_name)
    if regex:
        size = regex.group("size")
        return  (float(size) if '.' in size else int(size)) * int(regex.group("times") or 1)

    if default is not None:
        return default

    logging.warning(f"Could not infer model size from name '{model_name}'.")
    return default


def get_thinking_end_token_id(tokenizer: AutoTokenizer) -> int | None:
    think_id = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_id) == 1:
        return think_id[0]
    else: 
        logging.debug("Could not identify token id marking the end of thinking content.")
        return None
    

def supports_enable_thinking(tokenizer: AutoTokenizer) -> bool:
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return True
    except TypeError:
        return False
    

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
        
    raise ValueError(
        "Model name couldn't be matched. Please update the model-developer "
        "mapping in get_model_developer()."
    )