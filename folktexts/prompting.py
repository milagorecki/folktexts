"""Module to map risk-estimation questions to different prompting techniques.

e.g.,
- multiple-choice Q&A vs direct numeric Q&A;
- zero-shot vs few-shot vs CoT;
"""

from __future__ import annotations

import logging
# from functools import partial
from typing import Union

from copy import deepcopy
import inspect
import re

import pandas as pd
from transformers import AutoTokenizer

from .dataset import Dataset
from .qa_interface import QAInterface
from .task import TaskMetadata

from folktexts.acs.acs_columns_alt import simplified_value_maps
from folktexts.acs._utils import reset_cache
from folktexts.acs import (
    ACSTaskMetadata,
    ACS_TASK_DESCRIPTION,
    ACS_TASK_DESCRIPTION_DEFAULTS,
)
from folktexts.ts import (
    TableshiftBRFSSTaskMetadata,
    TABLESHIFT_TASK_DESCRIPTION,
    TABLESHIFT_TASK_DESCRIPTION_DEFAULTS,
)
from folktexts.sipp import(
    SIPPTaskMetadata,
    SIPP_TASK_DESCRIPTION,
    SIPP_TASK_DESCRIPTION_DEFAULTS,
)


SYSTEM_PROMPT = """\
You are a helpful assistant. You answer multiple-choice questions based on the information provided.
"""

# ACS_TASK_DESCRIPTION = """\
# The following data corresponds to a survey respondent. \
# The survey was conducted among US residents in 2018. \
# Please answer the question based on the information provided. \
# The data provided is enough to reach an approximate answer.
# """

# ACS_FEW_SHOT_TASK_DESCRIPTION = """\
# The following data corresponds to different survey respondents. \
# The survey was conducted among US residents in 2018. \
# Please answer each question based on the information provided. \
# The data provided is enough to reach an approximate answer for each person.
# """

ANTHROPIC_CHAT_PROMPT = """If had to select one of the options, my answer would be"""
GEMMA_CHAT_PROMPT = """The provided information suggests that the answer is"""

_valid_keys_cache = {}

DEFAULT_PROMPT_STYLE = {'format': 'bullet',
                        'connector': 'is',
                        'granularity': 'original',
                        'order': None,
                        'custom_prompt_prefix': None,
                        'custom_prompt_suffix': None
                        }


class PromptVariation:
    def __init__(
        self, description: str, task: ACSTaskMetadata | TableshiftBRFSSTaskMetadata
    ):
        assert isinstance(task, ACSTaskMetadata) or isinstance(
            task, TableshiftBRFSSTaskMetadata
        ) or isinstance(task,SIPPTaskMetadata), "Provide task object."
        self.description = description
        self.task = deepcopy(task)
        self.cache = {}

        # define how to apply the transformation (on each cell of a cell or row-wise)
        if hasattr(self, 'transform_row'):
            self._apply = self.transform_row
        elif hasattr(self, 'transform_feature'):
            self._apply = self._loop_over_features
        else:
            raise NotImplementedError("Subclass must define 'transform_row' or 'transform_feature'.")

    def __call__(self, row: pd.Series, **kwds) -> pd.Series:
        return self._apply(row, **kwds)

    def _loop_over_features(self, row: pd.Series, **kwds) -> pd.Series:
        # cast to object so string values can be assigned into numeric columns
        row = row.astype(object)
        # apply variation to every feature in the given row
        for col in row.index:
            if col in self.task.features:
                row[col] = self.transform_feature(col, row[col], **kwds)
        return row

    # def transform_feature(self, col, val):
    #     pass

    # def tranform_row():
    #     pass


class VaryFormat(PromptVariation):
    def __init__(self, task, format: str = DEFAULT_PROMPT_STYLE['format']):
        description = "Vary prompt format, default is 'bullet'."
        super().__init__(description, task)
        assert format in {
            "bullet",
            "comma",
            "text",
            "textbullet",
        }, "Currently only 'bullet', 'comma', 'text', 'textbullet' implemented."
        self.format = format
        logging.warning(
            "VaryFormat should be applied after the value mapping and adding the connector."
        )

    def transform_feature(self, col, val, **kwds):
        formats = {
            "bullet": lambda val: f"- {val}\n",
            "comma": lambda val: f"{val}, ",
            "text": lambda val: f"The {val}. ",
            "textbullet": lambda val: f"- The {val}.\n",
        }
        return formats[self.format](val)


class VaryConnector(PromptVariation):
    def __init__(self, task, connector: str = DEFAULT_PROMPT_STYLE['connector']):
        description = "Vary symbol used between feature name and value, default is 'is'."
        super().__init__(description, task)
        if connector == ":":
            self.connector = f"{connector} "
        else:
            self.connector = f" {connector} "
        logging.warning(
            "VaryConnector should be applied after value mapping has already been applied."
        )

    def transform_feature(self, col, val, **kwds):
        return f"{self.task.cols_to_text[col].short_description}{self.connector}{val}"


class VaryValueMap(PromptVariation):
    def __init__(self, task, granularity=DEFAULT_PROMPT_STYLE['granularity']):
        description = "Vary the granulariy of the feature map, default is original (higher granularity)."
        super().__init__(description, task)
        assert granularity in ["original", "low"]
        self.granularity = granularity
        self.reduced_granularity = False

    def transform_row(self, row: pd.Series, **kwds) -> pd.Series:
        if self.granularity == "low" and not self.reduced_granularity:
            logging.debug('Set col_to_text to lower granularity.')
            # empty cache of previously parsed pums codes to overwrite with new postprocessing for value map
            reset_cache()
            for col in self.task.features:
                if col in simplified_value_maps:
                    self.task.cols_to_text[col]._value_map = simplified_value_maps[col]
            self.reduced_granularity = True
        return self._loop_over_features(row, **kwds)

    def transform_feature(self, col, val, **kwds) -> pd.Series:
        # apply value map
        # TODO: can fail if val not in map, use get?
        return self.task.cols_to_text[col][val]


class VaryFeatureOrder(PromptVariation):
    def __init__(self, task, order: list | str = DEFAULT_PROMPT_STYLE['order']):
        description = "Vary the order of the features."
        super().__init__(description, task)
        if order:
            if isinstance(order, str):
                order = list(order.split(","))
            else:
                assert isinstance(
                    order, list
                ), "Expected order provided as list"  # mutable
            assert set(order) == set(
                self.task.features
            ), "Provide a complete ordering of all features"
        self.order = order

    def transform_row(self, row: pd.Series, **kwds):
        if not self.order:
            return row
        feature_set = set(self.task.features)  # for fast lookup
        # create iterator
        reordered_iter = iter(self.order)
        # reorder only feature-columns
        order_all = [
            next(reordered_iter) if col in feature_set else col for col in row.index
        ]
        return row[order_all]


class VaryPrefix(PromptVariation):

    def __init__(
        self,
        task: TaskMetadata,
        add_task_description: bool = True,
        custom_prompt_prefix: str = DEFAULT_PROMPT_STYLE['custom_prompt_prefix'],
        task_description: str = None,
    ):
        description = "Vary the prefix printed before the prompt, by default the task description is printed."
        super().__init__(description, task)
        if add_task_description:
            assert task_description is not None, "Provide a task description to add."
        self.task_description = task_description
        self.add_task_description = add_task_description
        self.custom_prefix = custom_prompt_prefix

    def transform_row(
        self,
        row: pd.Series,
        **kwds,
    ) -> pd.Series:
        if self.add_task_description:
            if self.custom_prefix:
                if self.custom_prefix[-1] != "\n":
                    self.custom_prefix = self.custom_prefix + "\n"
                prefix = self.task_description + self.custom_prefix
            else:
                prefix = self.task_description
            # add new line after prefix
            row = pd.Series(
                {
                    "_PREFIX": prefix + "\nInformation:\n",
                    **{index: val for index, val in row.items()},
                }
            )
        else:
            row = pd.Series(
                {
                    "_PREFIX": "Information:\n",
                    **{index: val for index, val in row.items()},
                }
            )
        return row


class VarySuffix(PromptVariation):
    def __init__(
        self, task, question: QAInterface = None, custom_prompt_suffix: str = DEFAULT_PROMPT_STYLE['custom_prompt_suffix'],
        skip_question: bool = False,
    ):
        description = "Vary the suffix, in particular the question."
        super().__init__(description, task)
        self.question = question if question else task.question
        self.custom_suffix = custom_prompt_suffix
        self.skip_question = skip_question

    def transform_row(
        self,
        row,
        **kwds,
    ):
        if self.skip_question:
            suffix = f"\n{self.question.get_answer_prefix()}{self.custom_suffix if self.custom_suffix else ''}"
        else:
            suffix = f"\n{self.question.get_question_prompt()}{self.custom_suffix if self.custom_suffix else ''}"
        row = pd.Series(
            {
                **{index: val for index, val in row.items()},
                "_SUFFIX": suffix,
            }
        )
        return row


def get_valid_keys(cls):
    if cls not in _valid_keys_cache:
        params = inspect.signature(cls.__init__).parameters
        _valid_keys_cache[cls] = set(params) - {"self", "args", "kwargs"}
    return _valid_keys_cache[cls]


_building_blocks_cache = {}
_last_cache_config = {}  # store the params used for the cache


def build_config_dict(
    task: TaskMetadata,
    question: QAInterface,
    add_task_description: bool,
    custom_prompt_prefix: str,
    custom_prompt_suffix: str,
    prompt_variation: dict | None,
):
    return {
        "task_name": task.name,
        "add_task_description": add_task_description,
        "custom_prompt_prefix": custom_prompt_prefix,
        "custom_prompt_suffix": custom_prompt_suffix,
        "prompt_variation": prompt_variation,
        "question": question,
    }


CONFIG_KEYS = ['task_name', 'add_task_description', 'custom_prompt_prefix',
               'custom_prompt_suffix', 'prompt_variation', 'question']
BLOCK_MAPPING = {
    'task_name': ['prefix'],
    'add_task_description': ['prefix'],
    'custom_prompt_prefix': ['prefix'],
    'custom_prompt_suffix': ['suffix'],
    'question': ['suffix'],
    'prompt_variation': ['prefix', 'suffix', 'order', 'granularity', 'connector', 'format'],
    # map indivdiual prompt variations
    'task_description': ['prefix'],
    'granularity': ['granularity'],
    'connector': ['connector'],
    'format': ['format'],
    'order': ['order'],
    'skip_question': ['suffix'],
}


def reset_building_block_cache():
    global _building_blocks_cache
    _building_blocks_cache.clear()


def update_building_blocks_if_needed(current_config, task):
    global _building_blocks_cache, _last_cache_config

    if len(_last_cache_config) == 0 or len(_building_blocks_cache) == 0:  # nothing cached yet
        changed_keys = CONFIG_KEYS
    else:
        changed_keys = []
        last_config = _last_cache_config or {}
        for key, value in current_config.items():
            if key == 'prompt_variation':  # is itself a dict
                all_varkeys = set(value.keys()).union(last_config.get(key, {}).keys())
                for varkey in all_varkeys:
                    if value.get(varkey) != last_config.get(key, {}).get(varkey):
                        prev = re.sub('\s+', ' ', str(last_config.get(key, {}).get(varkey))).strip()
                        curr = re.sub('\s+', ' ', str(value.get(varkey))).strip()
                        logging.debug(f"{varkey}(last -> curr): {prev} -> {curr}")
                        changed_keys.append(varkey)
            else:
                if value != last_config.get(key):
                    logging.debug(f"{key} (last -> curr): {last_config.get(key)} -> {value}")
                    changed_keys.append(key)

    affected_blocks = set()
    for key in changed_keys:
        affected_blocks.update(BLOCK_MAPPING.get(key, []))

    def _configure_variation(cls, default_kwargs):
        valid_keys = get_valid_keys(cls)
        # merge and overwrite defaults with variations
        merged = {**default_kwargs, **(current_config.get('prompt_variation', {}) or {})}
        # filter out keys not in class __init__
        filtered_kwargs = {k: v for k, v in merged.items() if k in valid_keys}
        return cls(task=task, **filtered_kwargs)
    
    task_description_dict = {"ACS": ACS_TASK_DESCRIPTION.substitute(ACS_TASK_DESCRIPTION_DEFAULTS), 
                             "BRFSS": TABLESHIFT_TASK_DESCRIPTION.substitute(TABLESHIFT_TASK_DESCRIPTION_DEFAULTS),
                             "SIPP": SIPP_TASK_DESCRIPTION.substitute(SIPP_TASK_DESCRIPTION_DEFAULTS)} 
    task_description =  next(task_description_dict[key] for key in task_description_dict.keys() if key in task.name)
    if len(affected_blocks) > 0:
        logging.info(f'Updating config for block: {affected_blocks}')
        if 'prefix' in affected_blocks:
            _building_blocks_cache['prefix'] = _configure_variation(
                    VaryPrefix,
                    {
                        "add_task_description": current_config['add_task_description'],
                        "custom_prompt_prefix": current_config['custom_prompt_prefix'],
                        "task_description": task_description,
                    },
                )
        if 'suffix' in affected_blocks:
            _building_blocks_cache['suffix'] = _configure_variation(
                    VarySuffix,
                    {
                        "question": current_config['question'],
                        "custom_prompt_suffix": current_config['custom_prompt_suffix'],
                    },
                )
        if 'order' in affected_blocks:
            _building_blocks_cache['order'] = _configure_variation(VaryFeatureOrder, {"order": None})
        if 'granularity' in affected_blocks:
            _building_blocks_cache['granularity'] = _configure_variation(VaryValueMap, {"granularity": "original"})
        if 'connector' in affected_blocks:
            _building_blocks_cache['connector'] = _configure_variation(VaryConnector, {"connector": "is"})
        if 'format' in affected_blocks:
            _building_blocks_cache['format'] = _configure_variation(VaryFormat, {"format": "textbullet"})

        _last_cache_config = current_config


def encode_row_prompt(
    row: pd.Series,
    task: TaskMetadata,
    question: QAInterface = None,
    custom_prompt_prefix: str = None,
    add_task_description: bool = True,
    custom_prompt_suffix: str = None,
    prompt_variation: dict | None = None,
) -> str:
    """Encode a question regarding a given row."""
    global _building_blocks_cache, _last_cache_config

    # ensure only feature defined for the task are used
    row = row[task.features]
    question = question or task.question

    # current config
    curr_config = build_config_dict(
        task, question, add_task_description, custom_prompt_prefix, custom_prompt_suffix, prompt_variation
    )

    # if update cache is different
    update_building_blocks_if_needed(current_config=curr_config, task=task)

    # order of value map (granularity), connector and format should not be changed
    for variation in ['prefix', 'suffix', 'order', 'granularity', 'connector', 'format']:
        if variation in _building_blocks_cache.keys():
            row = _building_blocks_cache[variation](row)
    return "".join(row.values)


def encode_row_prompt_few_shot(
    row: pd.Series,
    task: TaskMetadata,
    dataset: Dataset,
    n_shots: int,
    question: QAInterface = None,
    reuse_examples: bool = False,
    compose_few_shot_examples: Union[bool,list] = False,
    custom_prompt_prefix: str = None,
    prompt_variation: dict = {},
) -> str:
    """Encode a question regarding a given row using few-shot prompting.

    Parameters
    ----------
    row : pd.Series
        The row that the question will be about.
    task : TaskMetadata
        The task that the row belongs to.
    n_shots : int, optional
        The number of example questions and answers to use before prompting
        about the given row, by default 3.
    reuse_examples : bool, optional
        Whether to reuse the same examples for consistency. By default will
        resample new examples each time (`reuse_examples=False`).

    Returns
    -------
    prompt : str
        The encoded few-shot prompt.
    """
    logging.debug(f"Composition of few shot examples: {compose_few_shot_examples}")
    # Take `n_shots` random samples from the train set
    composition = compose_few_shot_examples if compose_few_shot_examples else "random"
    X_examples, y_examples = dataset.sample_n_train_examples(
        n_shots,
        reuse_examples=reuse_examples,
        composition=composition,
    )
    # Sorting examples by index to keep prompt fixed. Change by modifying the example order.
    X_examples = X_examples.sort_index()
    y_examples = y_examples.sort_index()
    logging.debug(f"ys  index:{y_examples.index.tolist()}")
    logging.debug(f"ys: {y_examples.values.tolist()}")

    prompt = ""

    # Get the question to ask
    question = question or task.question

    if prompt_variation and prompt_variation.get("example_order"):
        logging.debug("Varying the order of the examples.")
        print("Varying the order of the examples.")
        order_examples = list(prompt_variation.get("example_order", "").split(","))
        logging.debug(
            f"Received order: {order_examples} with length {len(order_examples)}, n_shots={n_shots}."
        )
        assert len(order_examples) == n_shots
        X_examples = X_examples.iloc[order_examples]
        y_examples = y_examples.iloc[order_examples]

    prompt_var = (prompt_variation.copy() or {})
    prompt_var.pop('example_order', 0)

    # Add `n` example rows with respective labels
    for i in range(n_shots):
        logging.debug(f"shot {i}: label = {question.get_answer_key_from_value(y_examples.iloc[i])}\t index = {y_examples.index[i]}")
        prompt += encode_row_prompt(
            X_examples.iloc[i],
            task=task,
            add_task_description=(i == 0),  # only add task description before first example
            custom_prompt_prefix=custom_prompt_prefix,
            prompt_variation={
                **prompt_var,
                "task_description": (
                    ACS_TASK_DESCRIPTION.substitute(
                        {
                            **ACS_TASK_DESCRIPTION_DEFAULTS,
                            "respondent": "different survey respondents",
                            "suffix": " for each person",
                        }
                    )
                    if task.name.startswith("ACS")
                    else TABLESHIFT_TASK_DESCRIPTION.substitute(
                        {
                            **TABLESHIFT_TASK_DESCRIPTION_DEFAULTS,
                            "respondent": "different survey respondents",
                            "suffix": " for each person",
                        }
                    )
                ),
                "custom_prompt_suffix": f" {question.get_answer_key_from_value(y_examples.iloc[i])}\n\n",
                "skip_question": True,
            },
        )

    # Add the target row without its label
    prompt += encode_row_prompt(
        row,
        task=task,
        add_task_description=False,
        custom_prompt_prefix=custom_prompt_prefix,
        question=question,
        prompt_variation=prompt_variation,
    )
    logging.debug(prompt)
    return prompt


def encode_row_prompt_chat(
    row: pd.Series,
    task: TaskMetadata,
    tokenizer: AutoTokenizer,
    question: QAInterface = None,
    **chat_template_kwargs,
) -> str:
    # TODO: implement two functions
    # - one for gemma-like models that are not compatible with system prompts
    # - and another for regular models compatible with system prompts
    logging.warning("NOTE :: Untested feature!!")

    return apply_chat_template(
        tokenizer,
        (SYSTEM_PROMPT + encode_row_prompt(row, task, question=question)),
        **chat_template_kwargs,
    )


def apply_chat_template(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str = None,
    chat_prompt: str = ANTHROPIC_CHAT_PROMPT,
    **kwargs,
) -> str:
    # Add system prompt
    conversation = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )

    # Add user prompt
    conversation.append({"role": "user", "content": user_prompt})

    # Using the Anthropic-style chat prompt
    conversation.append({"role": "assistant", "content": chat_prompt})

    # Default kwargs
    kwargs.setdefault("add_generation_prompt", False)

    # Apply prompt template
    filled_prompt = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        **kwargs,
    )

    # Make sure no special tokens follow the `CHAT_PROMPT`;
    # > some models add a newline character and/or a <end_of_turn> token
    filled_prompt = filled_prompt[: len(chat_prompt) + filled_prompt.find(chat_prompt)]
    return filled_prompt


# def encode_row_split(split: str | pd.DataFrame,
#                      task: str | ACSTaskMetadata | TableshiftBRFSSTaskMetadata,
#                      prompt_variation: dict = {'connector': 'is', 'format': 'text', 'granularity': 'original'},
#                      ):
#     from tqdm import tqdm
#     tqdm.pandas()
#     if isinstance(str, split):
#         assert split in ['test', 'train']
#         logging.info('Loading the dataset split..')
#         if isinstance(ACSTaskMetadata, task):
#             task = ACSTaskMetadata.get_task(task)
#             acs_dataset = ACSDataset.make_from_task(task=task, cache_dir=DATA_DIR)
#         elif isinstance(TableshiftBRFSSTaskMetadata, task):
#             pass
#     elif isinstance(split, pd.DataFrame):
#         data = split

#     encode_row = partial(encode_row_prompt, task=task, prompt_variation=prompt_variation)
#     data = data.progress_apply(lambda row: encode_row(row), axis=1).to_frame(name='text')

#     return data
