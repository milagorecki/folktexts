"""A collection of Tableshift BRFSS prediction tasks based on the tableshift package."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass
from string import Template

from tableshift.configs.benchmark_configs import BENCHMARK_CONFIGS, PreprocessorConfig
from tableshift.configs.experiment_config import ExperimentConfig
from tableshift.core.splitter import RandomSplitter
from tableshift.core.tasks import _TASK_REGISTRY, TaskConfig

from .._utils import hash_dict
from ..col_to_text import ColumnToText as _ColumnToText
from ..dataset import DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE
from ..qa_interface import DirectNumericQA, MultipleChoiceQA
from ..task import TaskMetadata
from ..threshold import Threshold
from . import brfss_columns, brfss_questions
from .tableshift_thresholds import brfss_diabetes_threshold, brfss_hypertension_threshold

TABLESHIFT_TASK_DESCRIPTION = Template("""\
The following data corresponds to $respondent. \
The survey was conducted among US residents in $year. \
Please answer the question based on the information provided. \
The data provided is enough to reach an approximate answer$suffix.
""")
TABLESHIFT_TASK_DESCRIPTION_DEFAULTS = {
    "respondent": "a survey respondent",
    "year": "the years 2015, 2017, 2019 and 2021",
    "suffix": "",
}

# custom preprocessor to avoid normalization and one-hot encoding
passthrough_preprocessor_config = PreprocessorConfig(
    categorical_features="passthrough",  # Options: one_hot (default), map_values, label_encode, passthrough.
    numeric_features="passthrough",  # Options: normalize (default), passthrough, map_values.
    domain_labels="label_encode",  # default
    passthrough_columns=["IYEAR"],
    dropna="rows",  # default
    use_extended_names=False,  # default
    map_targets=False,  # default
    cast_targets_to_default_type=False,  # default
    min_frequency=None,  # default
    max_categories=None,  # default
    n_bins=5,  # default
    sub_illegal_chars=True,  # default
)

# Map of BRFSS column names to ColumnToText objects
brfss_columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in brfss_columns.__dict__.values()
    if isinstance(col_mapper, _ColumnToText)
}


@dataclass
class TableshiftTask(TaskConfig, ExperimentConfig):
    """class merging info from task config and experiment config"""

    def __post_init__(self):
        self.name = self.tabular_dataset_kwargs["name"]


# TODO: check general tableshift compatibility
# should only depend on columns_map to include all possible columns of tableshift tasks + questions
@dataclass
class TableshiftBRFSSTaskMetadata(TaskMetadata):
    """A class to hold information on an Tableshift BRFSS prediction task."""

    # The tableshift task object from the folktables package
    tableshift_obj: TableshiftTask = None

    @classmethod
    def make_task(
        cls,
        name: str,
        features: list[str],
        target: str = None,
        sensitive_attribute: str = None,
        target_threshold: Threshold = None,
        multiple_choice_qa: MultipleChoiceQA = None,
        direct_numeric_qa: DirectNumericQA = None,
        description: str = None,
        tableshift_obj: TableshiftTask = None,
    ) -> TableshiftBRFSSTaskMetadata:
        """Create an Tableshift task object from the given parameters."""
        # Resolve target column name
        target_col_name = target_threshold.apply_to_column_name(target) if target_threshold is not None else target

        # Get default Q&A interfaces for this task's target column
        if multiple_choice_qa is None:
            multiple_choice_qa = brfss_questions.brfss_multiple_choice_qa_map.get(target_col_name)
        if direct_numeric_qa is None:
            direct_numeric_qa = brfss_questions.brfss_numeric_qa_map.get(target_col_name)

        return cls(
            name=name,
            features=features,
            target=target,
            cols_to_text=brfss_columns_map,
            sensitive_attribute=sensitive_attribute,
            target_threshold=target_threshold,
            multiple_choice_qa=multiple_choice_qa,
            direct_numeric_qa=direct_numeric_qa,
            description=description,
            tableshift_obj=tableshift_obj,
        )

    @classmethod
    def make_tableshift_task(
        cls,
        name: str,
        target_threshold: Threshold = None,
        description: str = None,
        val_size: float = DEFAULT_VAL_SIZE,  # default in getter
        test_size: float = DEFAULT_TEST_SIZE,  # default in getter
    ) -> TableshiftBRFSSTaskMetadata:

        # Get the task info/object from the tableshift package
        try:
            # get configs and create joint dataclass
            benchmark_configs = BENCHMARK_CONFIGS[
                name.lower()
            ]  # 'splitter', 'grouper', 'preprocessor_config', 'tabular_dataset_kwargs'
            # update to custom preprocessor config to avoid normalization and one-hot encoding
            benchmark_configs.preprocessor_config = passthrough_preprocessor_config

            default_splitter = copy.copy(benchmark_configs.splitter)
            benchmark_configs.splitter = RandomSplitter(
                val_size=val_size,
                random_state=default_splitter.random_state,
                test_size=test_size,
            )
            task_config = _TASK_REGISTRY[name.lower()]  # 'data_source_cls', 'feature_list'
            tableshift_task = TableshiftTask(**task_config.__dict__, **benchmark_configs.__dict__)

        except AttributeError:
            raise ValueError(f"Could not find task '{name.lower()}' in tableshift package.")

        logging.debug("Using only first sensitive attribute for task.")
        tableshift_task = cls.make_task(
            name=name,
            features=[
                f for f in tableshift_task.feature_list.names if f != tableshift_task.feature_list.target
            ],  # also includes feature used for domain shift
            target=tableshift_task.feature_list.target,
            sensitive_attribute=tableshift_task.grouper.features[0],
            target_threshold=target_threshold,
            description=description,
            tableshift_obj=tableshift_task,
        )

        return tableshift_task

    def __hash__(self) -> int:
        hashable_params = asdict(self)
        hashable_params.pop("cols_to_text")
        hashable_params.pop("tableshift_obj")
        return int(hash_dict(hashable_params), 16)


# Instantiate folktables tasks
tableshift_brfss_diabetes_task = TableshiftBRFSSTaskMetadata.make_tableshift_task(
    name="BRFSS_Diabetes",
    target_threshold=brfss_diabetes_threshold,
    description="predict whether an individual has ever been told they have diabetes",
)

tableshift_brfss_hypertension_task = TableshiftBRFSSTaskMetadata.make_tableshift_task(
    name="BRFSS_Blood_Pressure",
    target_threshold=brfss_hypertension_threshold,
    description="predict whether an individual has ever been told they have diabetes",
)
