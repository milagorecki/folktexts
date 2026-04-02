"""
BRFSS task columns

Value Maps adapted from https://github.com/mlfoundations/tableshift/blob/main/tableshift/datasets/brfss.py

For more information on datasets and access in TableShift, see:
* https://tableshift.org/datasets.html
* https://github.com/mlfoundations/tableshift

Accessed via https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system.
Raw Data: https://www.cdc.gov/brfss/annual_data/annual_data.htm
Data Dictionary: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf

"""

# from functools import partial
# from pathlib import Path
import logging

from ..col_to_text import ColumnToText
from ..qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from .tableshift_thresholds import (
    brfss_diabetes_threshold,
    brfss_hypertension_threshold,
)

tableshift_physical_health = ColumnToText(
    name="PHYSHLTH",
    short_description="number of days during the past 30 days where physical health was not good",
    value_map=lambda x: f"{int(x)} days",
    missing_value_fill="N/A (refused or unknown)",
)

logging.debug(
    "Value map encoding following tableshift encoding."
    " Note, that inconsistent with other features, 1.0 corresponds to 'No' and 2.0 corresponds to 'Yes'."
)
tableshift_high_blood_pressure = ColumnToText(
    name="HIGH_BLOOD_PRESS",
    short_description="prior diagnosis of high blood pressure",
    value_map={
        1.0: "No",
        2.0: "Yes",
    },
    missing_value_fill="N/A (refused or unknown)",
)

# HYPERTENSION question
brfss_hypertension_qa = MultipleChoiceQA(
    column=brfss_hypertension_threshold.apply_to_column_name("HIGH_BLOOD_PRESS"),
    text="Has this person ever been told to have high blood pressure by a health professional?",
    choices=(
        Choice("Yes, this person has been told they have high blood pressure", 1),
        Choice("No, this person has not been told they have high blood pressure", 0),
    ),
)

brfss_hypertension_numeric_qa = DirectNumericQA(
    column=brfss_hypertension_threshold.apply_to_column_name("HIGH_BLOOD_PRESS"),
    text=(
        "What is the probability that this person has ever been told"
        " they have high blood pressure by a health professional?"
    ),
)

tableshift_hypertension_target_col = ColumnToText(
    name=brfss_hypertension_threshold.apply_to_column_name("HIGH_BLOOD_PRESS"),
    short_description="prior diagnosis of diabetes",
    value_map={
        0.0: "No",
        1.0: "Yes",
    },
    missing_value_fill="N/A (refused or unknown)",
    question=brfss_hypertension_qa,
)

tableshift_chol_chk_past_5_years = ColumnToText(
    name="CHOL_CHK_PAST_5_YEARS",
    short_description="time since last blood cholesterol check",
    value_map={
        1.0: "never checked",
        2.0: "within the past year (anytime less than 12 months ago)",
        3.0: "within the past 2 years (1 year but less than 2 years ago)",
        4.0: "within the past 5 years (2 years but less than 5 years ago)",
        5.0: "5 or more years ago",
        # 7: 'don\'t know',
        # 9: 'refused'
    },
    missing_value_fill="N/A (refused or unknown)",
)

logging.debug("Encoding value map keys of 'TOLDHI' as strings.")
tableshift_told_hi = ColumnToText(
    name="TOLDHI",
    short_description="prior diagnosis of high blood cholesterol",
    value_map={
        "1.0": "Yes",
        "2.0": "No",
        # 7: 'don\'t know',
        # 9: 'refused',
        "NOTASKED_MISSING": "N/A (answer missing, because question was not asked)",
    },
    missing_value_fill="N/A (refused, unknown or not asked)",
)

tableshift_bmi5 = ColumnToText(
    name="BMI5",
    short_description="computed body mass index (kg/m^2)",
    value_map=lambda x: f"{x:.2f}",
    missing_value_fill="N/A (refused or unknown)",
)

tableshift_bmi5cat = ColumnToText(
    name="BMI5CAT",
    short_description="body mass index (kg/m^2) category",
    value_map={
        1.0: "underweight",  # (BMI < 1850)",
        2.0: "normal weight",  # (1850 <= BMI < 2500)",
        3.0: "overweight",  # (2500 <= BMI < 3000)",
        4.0: "obese",  # (3000 <= BMI < 9999)",
    },
    missing_value_fill="N/A (missing)",
)

tableshift_smoke100 = ColumnToText(
    name="SMOKE100",
    short_description="history of smoking at least 100 cigarettes in their lifetime",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

logging.debug("Encoding value map keys of 'SMOKDAY2' as strings.")
tableshift_smokday2 = ColumnToText(
    name="SMOKDAY2",
    short_description="current frequency of cigarette smoking",  # every day, some days, or not at all",
    value_map={
        "1.0": "every day",
        "2.0": "some days",
        "3.0": "not at all",
        # 7: 'don\'t know',
        # 9: 'refused',
        "NOTASKED_MISSING": "answer missing, because question was not asked",
    },
    missing_value_fill="N/A (refused, unknown or not asked)",
)

tableshift_cvdstrk3 = ColumnToText(
    name="CVDSTRK3",
    short_description="previous stroke event",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },  # TODO: Check if mapping should be changed to 0,1
    missing_value_fill="N/A (refused or unknown)",
)

tableshift_michd = ColumnToText(
    name="MICHD",
    short_description="previous event of a myocardial infarction (MI) or coronary heart disease (CHD)",
    value_map={
        1.0: "Yes, reported myocardial infarction or coronary heart disease.",
        2.0: "No, did not report myocardial infarction or coronary heart disease.",
    },  # TODO: Check if mapping should be changed to 0,1
    missing_value_fill="N/A (missing)",
)

tableshift_fruit_once_per_day = ColumnToText(
    name="FRUIT_ONCE_PER_DAY",
    short_description="consumption of fruit one or more times per day",
    value_map={
        1.0: "Yes",  # , consumed fruit one or more times per day
        2.0: "No",  # , consumed fruit less than one time per day.
    },
    missing_value_fill="N/A (refused, unknown or missing)",
)

tableshift_veg_once_per_day = ColumnToText(
    name="VEG_ONCE_PER_DAY",
    short_description="consumption of vegetables one or more times per day",
    value_map={
        1.0: "Yes",  # , consumed vegetables one or more times per day
        2.0: "No",  # , consumed vegetables less than one time per day
    },
    missing_value_fill="N/A (refused, unknown or missing)",
)

tableshift_drnk_per_week = ColumnToText(
    name="DRNK_PER_WEEK",
    short_description="total number of alcoholic beverages consumed per week",
    value_map=lambda x: f"{int(x)} alcoholic beverages per week",
    missing_value_fill="N/A (missing)",
)

tableshift_rfbing5 = ColumnToText(
    name="RFBING5",
    short_description=(
        "binge drinking behaviour (i.e. >= 5 drinks per occasion for males, >= 4 drinks per occasion for females)"
    ),
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused, unknown or missing)",
)

tableshift_totinda = ColumnToText(
    name="TOTINDA",
    short_description="leisure-time physical activity in the past 30 days",
    value_map={
        1.0: "Yes, had physical activity or exercise during the past 30 days other than regular job.",
        2.0: "No physical activity or exercise during the past 30 days other than regular job",
    },
    missing_value_fill="N/A (missing)",
)

tableshift_income = ColumnToText(
    name="INCOME",
    short_description="total annual household income",
    value_map={
        1.0: "Less than $10,000",
        2.0: "$10,000 to less than $15,000",
        3.0: "$15,000 to less than $20,000",
        4.0: "$20,000 to less than $25,000",
        5.0: "$25,000 to less than $35,000",
        6.0: "$35,000 to less than $50,000",
        7.0: "$50,000 to less than $75,000",
        8.0: "$75,000 or more (BRFSS 2015-2019) or $75,000 to less than $100,000 (BRFSS 2021)",
        9.0: "$100,000 to less than $150,000",
        10.0: "$150,000 to less than $200,000",
        11.0: "$200,000  or more",
    },
    missing_value_fill="N/A (refused or unknown)",
)

tableshift_marital = ColumnToText(
    name="MARITAL",
    short_description="marital status",
    value_map={
        1.0: "Married",
        2.0: "Divorced",
        3.0: "Widowed",
        4.0: "Separated",
        5.0: "Never married",
        6.0: "Member of an unmarried couple",
    },
    missing_value_fill="N/A (refused)",
)


tableshift_checkup1 = ColumnToText(
    name="CHECKUP1",
    short_description="time since last visit to a doctor for a general checkup",
    value_map={
        1.0: "Within past year (anytime < 12 months ago)",
        2.0: "Within past 2 years (1 year but < 2 years ago)",
        3.0: "Within past 5 years (2 years but < 5 years ago)",
        4.0: "5 or more years ago",
        # 7: 'Don’t know/Not sure',
        8.0: "Never",
        # 9: 'Refused',
    },
    missing_value_fill="N/A (refused or unknown)",
)

tableshift_educa = ColumnToText(
    name="EDUCA",
    short_description="highest grade or year of school completed",
    value_map={
        1.0: "Never attended school or only kindergarten",
        2.0: "Grades 1 through 8 (Elementary)",
        3.0: "Grades 9 through 11 (Some high school)",
        4.0: "Grade 12 or GED (High school graduate)",
        5.0: "College 1 year to 3 years (Some college or technical school)",
        6.0: "College 4 years or more (College graduate)",
        # 9: 'Refused'
    },
    missing_value_fill="N/A (refused)",
)

tableshift_health_cov = ColumnToText(
    name="HEALTH_COV",
    short_description="current health coverage status",
    value_map={
        1.0: "Yes, has a health care coverage",
        2.0: "No, does not have health care coverage",
        9.0: "Person below 18 years old or above 64 years old or answer refused, unknown or missing",
    },
    missing_value_fill="N/A (below 18 years old, above 64 years old or refused, unknown or missing)",
)

tableshift_menthlth = ColumnToText(
    name="MENTHLTH",
    short_description="number of days mental health was not good in the past 30 days",
    value_map=lambda x: f"{int(x)} days",
    missing_value_fill="N/A (refused or unknown)",
)

tableshift_iyear = ColumnToText(
    name="IYEAR",
    short_description="year of survey",
    value_map=lambda x: f"{int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)


def parse_state(val):
    state_dict = {
        1.0: "Alabama",
        4.0: "Arizona",
        5.0: "Arkansas",
        6.0: "California",
        8.0: "Colorado",
        9.0: "Connecticut",
        10.0: "Delaware",
        11.0: "District of Columbia",
        12.0: "Florida",
        13.0: "Georgia",
        15.0: "Hawaii",
        16.0: "Idaho",
        17.0: "Illinois ",
        18.0: "Indiana",
        19.0: "Iowa",
        20.0: "Kansas",
        21.0: "Kentucky",
        22.0: "Louisiana ",
        23.0: "Maine",
        24.0: "Maryland",
        25.0: "Massachusetts",
        26.0: "Michigan",
        27.0: "Minnesota",
        28.0: "Mississippi",
        29.0: "Missouri",
        30.0: "Montana",
        31.0: "Nebraska",
        32.0: "Nevada",
        33.0: "New Hampshire",
        34.0: "New Jersey",
        35.0: "New Mexico",
        36.0: "New York",
        37.0: "North Carolina",
        38.0: "North Dakota",
        39.0: "Ohio",
        40.0: "Oklahoma",
        41.0: "Oregon",
        42.0: "Pennsylvania",
        44.0: "Rhode Island",
        45.0: "South Carolina",
        46.0: "South Dakota",
        47.0: "Tennessee",
        48.0: "Texas",
        49.0: "Utah",
        50.0: "Vermont",
        51.0: "Virginia",
        53.0: "Washington",
        54.0: "West Virginia",
        55.0: "Wisconsin",
        56.0: "Wyoming",
        66.0: "Guam",
        72.0: "Puerto Rico",
    }
    if val not in state_dict.keys():
        logging.debug("Could not find FIPS code for state in dictionary.")
        return "N/A"
    else:
        return state_dict[int(val)]


tableshift_state = ColumnToText(
    name="STATE",
    short_description="state of residence",
    value_map=parse_state,
    missing_value_fill="N/A (refused or unknown)",
)

tableshift_medcost = ColumnToText(
    name="MEDCOST",
    short_description="unmet medical need due to costs in the last 12 months",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

logging.debug("PRACE1 encoded with mix of integers and floats due to grouping by some tasks, inconsistent to rest.")
tableshift_prace1 = ColumnToText(
    name="PRACE1",
    short_description="preferred race category",
    value_map={
        0: "Non-White",  # added for BRFSS Blood Pressure (Grouper turns race into int)
        1: "White",  # added for BRFSS Blood Pressure (Grouper turns race into int)
        1.0: "White",  # noqa: F601
        2.0: "Black or African American",
        3.0: "American Indian or Alaskan Native",
        4.0: "Asian",
        5.0: "Native Hawaiian or other Pacific Islander",
        6.0: "Other race",
        7.0: "No preferred race",  # na_value
        8.0: "Multiracial, but preferred race not answered",  # na_value
        77.0: "Don’t know/Not sure",  # na_value
        9.0: "Refused",
    },
    missing_value_fill="N/A (refused or no preferred race)",
)

logging.debug("SEX encoded with integers 0/1, inconsistent to rest.")
tableshift_sex = ColumnToText(
    name="SEX",
    short_description="gender",
    value_map={
        0: "Male",
        1: "Female",
    },
    missing_value_fill="N/A (refused or unknown)",
)

logging.debug("DIABETES recoded for BRFSS Diabetes Task to a binary indicator, apdapt if using for another task.")
tableshift_diabetes = ColumnToText(
    name="DIABETES",
    short_description="prior diagnosis of diabetes",
    value_map={
        1.0: "Yes",
        2.0: "Yes, but female told only during pregnancy",
        3.0: "No",
        4.0: "No, pre-diabetes or borderline diabetes",
        # 7.0: "Don’t know/Not Sure",
        # 9.0: "Refused, BLANK Not asked or Missing",
    },
    missing_value_fill="N/A (refused or unknown)",
)

# DIABETES question
brfss_diabetes_qa = MultipleChoiceQA(
    column=brfss_diabetes_threshold.apply_to_column_name("DIABETES"),
    text="Has this person ever been diagnosed with diabetes?",
    choices=(
        Choice("Yes, this person has been told they have diabetes", 1),
        Choice("No, this person has not been told they have diabetes", 0),
    ),
)

brfss_diabetes_numeric_qa = DirectNumericQA(
    column=brfss_diabetes_threshold.apply_to_column_name("DIABETES"),
    text=("What is the probability that this person has ever been told they have diabetes?"),
)

tableshift_diabetes_target_col = ColumnToText(
    name=brfss_diabetes_threshold.apply_to_column_name("DIABETES"),
    short_description="ever told to have diabetes",
    value_map={
        0: "No",
        1: "Yes",
    },
    missing_value_fill="N/A (refused or unknown)",
    question=brfss_diabetes_qa,
)


def parse_age_group(val):
    age_dict = {
        1.0: "18-24",
        2.0: "25-29",
        3.0: "30-34",
        4.0: "35-39",
        5.0: "40-44",
        6.0: "45-49",
        7.0: "50-54",
        8.0: "55-59",
        9.0: "60-61",
        10.0: "62-64",
        11.0: "65-66",
        12.0: "67-69",
        13.0: "70-74",
        14.0: "75-79",
        15.0: "80-84",
        16.0: "85 or",
    }
    return f"{age_dict[val]} years old"


tableshift_age_group = ColumnToText(
    name="AGEG5YR",
    short_description="age group (in intervals of 5 years)",
    value_map=parse_age_group,
    missing_value_fill="N/A (refused or unknown)",
)

tableshift_chcscncr = ColumnToText(
    name="CHCSCNCR",
    short_description="prior diagnosis of skin cancer",
    value_map={
        1.0: "Yes",
        2.0: "No",
        # 7.0: 'unknown or unsure',
        # 9.0: 'refused',
    },
    missing_value_fill="N/A (refused or unknown)",
)

tableshift_chcocncr = ColumnToText(
    name="CHCOCNCR",
    short_description="prior diagnosis of other cancer than skin cancer",
    value_map={
        1.0: "Yes",
        2.0: "No",
        # 7.0: 'unknown or unsure',
        # 9.0: 'refused',
    },
    missing_value_fill="N/A (refused or unknown)",
)


logging.debug(
    "POVERTY columns gets overwritten in tableshift preprocessing using INCOME "
    "and a slightly lower threshold of $25,000."
)
tableshift_poverty = ColumnToText(
    name="POVERTY",
    short_description=(
        "binary indicator of whether individual's income falls below 2021 poverty guideline for a family of four"
    ),
    value_map={
        1: "Yes",
        0: "No",
    },
)

tableshift_employ1 = ColumnToText(
    name="EMPLOY1",
    short_description="current employment status",
    value_map={
        1.0: "Employed for wages",
        2.0: "Self-employed",
        3.0: "Out of work for 1 year or more",
        4.0: "Out of work for less than 1 year",
        5.0: "Homemaker",
        6.0: "Student",
        7.0: "Retired",
        8.0: "Unable to work",
        # 9.0: "Refused",
    },
    missing_value_fill="N/A (refused)",
)
