"""A collection of Tableshift BRFSS prediction tasks based on the tableshift package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy

from .._utils import hash_dict
from ..col_to_text import ColumnToText as _ColumnToText
from ..qa_interface import DirectNumericQA, MultipleChoiceQA
from ..task import TaskMetadata
from ..threshold import Threshold
from . import sipp_columns, sipp_questions

import logging
from string import Template

SIPP_TASK_DESCRIPTION = Template("""\
The following data corresponds to $respondent. \
The survey was conducted among US residents in $year. \
Please answer the question based on the information provided. \
The data provided is enough to reach an approximate answer$suffix.
""")
SIPP_TASK_DESCRIPTION_DEFAULTS = {
    "respondent": "a survey respondent",
    "year": "year 2014",
    "suffix": "",
}

# Map of SIPP column names to ColumnToText objects
sipp_columns_map = sipp_questions.sipp_columns_map

@dataclass
class SIPPTaskMetadata(TaskMetadata):
    """A class to hold information on an Tableshift BRFSS prediction task."""

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
    ) -> SIPPTaskMetadata:
        """Create an Tableshift task object from the given parameters."""
        # Resolve target column name
        target_col_name = (
            target_threshold.apply_to_column_name(target)
            if target_threshold is not None
            else target
        )

        # Get default Q&A interfaces for this task's target column
        if multiple_choice_qa is None:
            multiple_choice_qa = sipp_questions.sipp_multiple_choice_qa_map.get(
                target_col_name
            )
        if direct_numeric_qa is None:
            direct_numeric_qa = sipp_questions.sipp_numeric_qa_map.get(
                target_col_name
            )

        return cls(
            name=name,
            features=features,
            target=target,
            cols_to_text=sipp_columns_map,
            sensitive_attribute=sensitive_attribute,
            target_threshold=target_threshold,
            multiple_choice_qa=multiple_choice_qa,
            direct_numeric_qa=direct_numeric_qa,
            description=description,
        )

    @classmethod
    def make_sipp_task(
        cls,
        name: str,
        target_threshold: Threshold = None,
        description: str = None,
    ) -> SIPPTaskMetadata:
        
        target_col = "OPM_RATIO"

        sipp_task = cls.make_task(
            name=name,
            features=[
                col.name
                for col in sipp_columns_map.values() if target_col not in col.name
            ],  # also includes feature used for domain shift
            target=target_col,
            sensitive_attribute='GENDER',
            target_threshold=target_threshold,
            description=description,
        )

        return sipp_task

    def __hash__(self) -> int:
        hashable_params = asdict(self)
        hashable_params.pop("cols_to_text")
        return int(hash_dict(hashable_params), 16)


# Instantiate task
sipp_task = SIPPTaskMetadata.make_sipp_task(
    name="SIPP",
    target_threshold=sipp_columns.sipp_threshold,
    description="predict wether a household's income is at least three times the poverty threshold",
)

