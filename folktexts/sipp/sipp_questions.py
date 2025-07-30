"""A collection of instantiated ACS column objects and ACS tasks."""

from __future__ import annotations

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import DirectNumericQA as _DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA as _MultipleChoiceQA

from . import sipp_columns

# Map of SIPP column names to ColumnToText objects
sipp_columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in sipp_columns.__dict__.values()
    if isinstance(col_mapper, ColumnToText)
}

# Map of numeric SIPP questions
sipp_numeric_qa_map: dict[str, object] = {
    question.column: question
    for question in sipp_columns.__dict__.values()
    if isinstance(question, _DirectNumericQA)
}

# Map of multiple-choice SIPP questions
sipp_multiple_choice_qa_map: dict[str, object] = {
    question.column: question
    for question in sipp_columns.__dict__.values()
    if isinstance(question, _MultipleChoiceQA)
}

# ... include all multiple-choice questions defined in the column descriptions
sipp_multiple_choice_qa_map.update(
    {
        col_to_text.name: col_to_text.question
        for col_to_text in sipp_columns_map.values()
        if (isinstance(col_to_text, ColumnToText) and col_to_text._question is not None)
    }
)
