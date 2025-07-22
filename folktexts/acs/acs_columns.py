"""Module to hold ACS column mappings from values to natural text.
"""

from functools import partial
from pathlib import Path

from ..col_to_text import ColumnToText
from ..qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from ._utils import parse_pums_code
from .acs_thresholds import (
    acs_employment_threshold,
    acs_health_insurance_threshold,
    acs_income_threshold,
    acs_mobility_threshold,
    acs_poverty_ratio_threshold,
    acs_public_coverage_threshold,
    acs_travel_time_threshold,
)

# Path to ACS codebook files
ACS_CODEBOOK_DIR = Path(__file__).parent / "data"

# AGEP: Age
acs_age = ColumnToText(
    "AGEP",
    short_description="age",
    value_map=lambda x: f"{int(x)} years old",
    verbalize=lambda x: f"The person is {x}.",
)

# COW: Class of Worker
acs_class_of_worker = ColumnToText(
    "COW",
    short_description="class of worker",
    value_map={
        1: "Working for a for-profit private company or organization",
        2: "Working for a non-profit organization",
        3: "Working for the local government",
        4: "Working for the state government",
        5: "Working for the federal government",
        6: "Owner of non-incorporated business, professional practice, or farm",
        7: "Owner of incorporated business, professional practice, or farm",
        8: "Working without pay in a for-profit family business or farm",
        9: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    verbalize=lambda x: f"The person is {x}.",
)

# SCHL: Educational Attainment
acs_schooling = ColumnToText(
    "SCHL",
    short_description="highest educational attainment",
    value_map={
        1: "N/A - no schooling completed",
        2: "Nursery school / preschool",
        3: "Kindergarten",
        4: "1st grade only",
        5: "2nd grade",
        6: "3rd grade",
        7: "4th grade",
        8: "5th grade",
        9: "6th grade",
        10: "7th grade",
        11: "8th grade",
        12: "9th grade",
        13: "10th grade",
        14: "11th grade",
        15: "12th grade, no diploma",
        16: "Regular high school diploma",
        17: "GED or alternative credential",
        18: "Some college, less than 1 year",
        19: "Some college, 1 or more years, no degree",
        20: "Associate's degree",
        21: "Bachelor's degree",
        22: "Master's degree",
        23: "Professional degree beyond a bachelor's degree",
        24: "Doctorate degree",
    },
    verbalize=lambda x: f"The person is {'the' if x.startswith('Owner') else ''} {x.lower()}.",
)

# MAR: Marital Status
acs_marital_status = ColumnToText(
    "MAR",
    short_description="marital status",
    value_map={
        1: "Married",
        2: "Widowed",
        3: "Divorced",
        4: "Separated",
        5: "Never married",
    },
    verbalize=lambda x: f"The person {'was' if x.startswith('Never') else 'is'} {x.lower()}.",
)

# OCCP: Occupation
acs_occupation = ColumnToText(
    "OCCP",
    short_description="occupation",
    value_map=partial(
        parse_pums_code,
        file=ACS_CODEBOOK_DIR / "OCCP.txt",
        postprocess=lambda x: x.split("-", 1)[1] if len(x.split("-", 1)) == 2 else x.split("-", 1)[0],
    ),
    verbalize=lambda x: f"The person's occupation is '{x}'.",
)

# POBP: Place of Birth
acs_place_of_birth = ColumnToText(
    "POBP",
    short_description="place of birth",
    value_map=partial(
        parse_pums_code,
        file=ACS_CODEBOOK_DIR / "POBP.txt",
        postprocess=lambda x: (x[: x.find("/")] if "/" in x else x).strip(),
    ),
    verbalize=lambda x: f"The person is born in {x}.",
)

# RELP: Relationship to Reference Person
acs_relationship = ColumnToText(
    "RELP",
    short_description="relationship to the reference person in the survey",
    value_map={
        0: "The reference person itself",
        1: "Husband/wife",
        2: "Biological son or daughter",
        3: "Adopted son or daughter",
        4: "Stepson or stepdaughter",
        5: "Brother or sister",
        6: "Father or mother",
        7: "Grandchild",
        8: "Parent-in-law",
        9: "Son-in-law or daughter-in-law",
        10: "Other relative",
        11: "Roomer or boarder",
        12: "Housemate or roommate",
        13: "Unmarried partner",
        14: "Foster child",
        15: "Other non-relative",
        16: "Institutionalized group quarters population",
        17: "Non-institutionalized group quarters population",
    },
    verbalize=lambda x: f"Relative to the reference person in the survey, the person is {'the ' if not (x.startswith('The') or x.endswith('population')) else 'part of a ' if x.endswith('population') else ''}{x.lower()}.",
)

# WKHP: Usual Hours Worked per Week
acs_work_hours = ColumnToText(
    "WKHP",
    short_description="usual number of hours worked per week",
    missing_value_fill="N/A (less than 16 years old, or did not work during the past 12 months)",
    value_map=lambda x: f"{int(x)} hours",
    verbalize=lambda x: (
        f"Usually, the person works {x} per week."
        if not x.startswith("N/A")
        else "The person is less that 16 years old or did not work during the past 12 months."
    ),
)

# SEX: Sex
acs_sex = ColumnToText(
    "SEX",
    short_description="sex",
    value_map={
        1: "Male",
        2: "Female",
    },
    verbalize=lambda x: f"The person identifies as {x.lower()}.",
)

# RAC1P: Race
acs_race = ColumnToText(
    "RAC1P",
    short_description="race",
    value_map={
        1: "White",
        2: "Black or African American",
        3: "American Indian",
        4: "Alaska Native",
        5: (
            "American Indian and Alaska Native tribes specified; or American "
            "Indian or Alaska Native, not specified and no other races"
        ),
        6: "Asian",
        7: "Native Hawaiian and Other Pacific Islander",
        8: "Some other race alone (non-White)",
        # 8: "Some other race alone",
        9: "Two or more races",
    },
    verbalize=lambda x: f"The person identifies {'as ' if not(x.startswith('Some') or  x.startswith('Two')) else 'with '}{x}.",
)

acs_race_native_hawaiian= ColumnToText(
    "RACNH", 
    short_description="Native Hawaiian",
    value_map={
        0: "No",
        1: "Yes",
    },
    verbalize=lambda x: f"The person {'identifies' if x=='Yes' else 'does not indetify'} as Native Hawaiian.",
)

acs_race_black= ColumnToText(
    "RACBLK", 
    short_description="Black or African American",
    value_map={
        0: "No",
        1: "Yes",
    },
    verbalize=lambda x: f"The person {'identifies' if x=='Yes' else 'does not indetify'} as Black or African American.",
)

acs_race_asian= ColumnToText(
    "RACASN", 
    short_description="Asian",
    value_map={
        0: "No",
        1: "Yes",
    },
    verbalize=lambda x: f"The person {'identifies' if x=='Yes' else 'does not indetify'} as Asian.",
)

acs_race_white= ColumnToText(
    "RACWHT", 
    short_description="White",
    value_map={
        0: "No",
        1: "Yes",
    },
    verbalize=lambda x: f"The person {'identifies' if x=='Yes' else 'does not indetify'} as White.",
)

acs_race_pi= ColumnToText(
    "RACPI", 
    short_description="Pacific Islander other than Native Hawaiian",
    value_map={
        0: "No",
        1: "Yes",
    },
    verbalize=lambda x: f"The person {'identifies' if x=='Yes' else 'does not indetify'} as Pacific Islander other than Native Hawaiian.",
)

acs_race_other= ColumnToText(
    "RACSOR", 
    short_description="Some other race",
    value_map={
        0: "No",
        1: "Yes",
    },
    verbalize=lambda x: f"The person {'identifies' if x=='Yes' else 'does not indetify'} with some other race.",
)

acs_race_aian= ColumnToText(
    "RACAIAN", 
    short_description="American Indian and Alaska Native",
    value_map={
        0: "No",
        1: "Yes",
    },
    verbalize=lambda x: f"The person {'identifies' if x=='Yes' else 'does not indetify'} as American Indian and Alaska Native.",
)

# PINCP: Yearly Income
acs_income = ColumnToText(
    "PINCP",
    short_description="yearly income",
    missing_value_fill="N/A (less than 15 years old)",
    value_map=lambda x: f"${int(x):,}",
    verbalize=lambda x: (
        f"The person has a yearly income of {x}."
        if not x.startswith("N/A")
        else "The person is less that 15 years old."
    ),
)

# PINCP: Yearly Income (Thresholded)
acs_income_qa = MultipleChoiceQA(
    column=acs_income_threshold.apply_to_column_name("PINCP"),
    text="What is this person's estimated yearly income?",
    choices=(
        Choice("Below $50,000", 0),
        Choice("Above $50,000", 1),
    ),
)

acs_income_numeric_qa = DirectNumericQA(
    column=acs_income_threshold.apply_to_column_name("PINCP"),
    text=(
        "What is the probability that this person's estimated yearly income is "
        "above $50,000 ?"
    ),
)

acs_income_target_col = ColumnToText(
    name=acs_income_threshold.apply_to_column_name("PINCP"),
    short_description="yearly income",
    missing_value_fill="N/A (less than 15 years old)",
    question=acs_income_qa,
    verbalize=lambda x: (
        f"The person has a yearly income of {x}."
        if not x.startswith("N/A")
        else "The person is less that 15 years old."
    ),
)

# PUBCOV: Public Health Coverage (Original)
acs_pubcov_og_qa = MultipleChoiceQA(
    column="PUBCOV",
    text="Does this person have public health insurance coverage?",
    choices=(
        Choice("Yes, person is covered by public health insurance", 1),
        Choice(
            "No, person is not covered by public health insurance", 2
        ),  # NOTE: value=2 for no public coverage!
    ),
)

acs_pubcov = ColumnToText(
    "PUBCOV",
    short_description="public health coverage status",
    value_map={
        1: "Covered by public health insurance",
        2: "Not covered by public health insurance",
    },
    question=acs_pubcov_og_qa,
    verbalize=lambda x: f"The person is {x.lower()}.",
)

# PUBCOV: Public Health Coverage (Thresholded)
acs_pubcov_qa = MultipleChoiceQA(
    column=acs_public_coverage_threshold.apply_to_column_name("PUBCOV"),
    text="Does this person have public health insurance coverage?",
    choices=(
        Choice("Yes, person is covered by public health insurance", 1),
        Choice(
            "No, person is not covered by public health insurance", 0
        ),  # NOTE: value=0 for no public coverage!
    ),
)

acs_pubcov_numeric_qa = DirectNumericQA(
    column=acs_public_coverage_threshold.apply_to_column_name("PUBCOV"),
    text=(
        "What is the probability that this person is covered by public health insurance?"
    ),  # NOTE: value=1 for yes, 0 for no
)

acs_pubcov_target_col = ColumnToText(
    name=acs_public_coverage_threshold.apply_to_column_name("PUBCOV"),
    short_description="public health coverage status",
    question=acs_pubcov_qa,
    use_value_map_only=True,
    verbalize=lambda x: f"The person is {x.lower()}.",
)

# DIS: Disability Status
acs_disability = ColumnToText(
    "DIS",
    short_description="disability status",
    value_map={
        1: "With a disability",
        2: "No disability",
    },
    verbalize=lambda x: f"The person has {'a' if x.startswith('With') else 'no'} disability.",
)

# ESP: Employment Status of Parents
acs_emp_parents = ColumnToText(
    "ESP",
    short_description="employment status of parents",
    value_map={
        1: "Living with two parents, both employed",
        2: "Living with two parents, only Father is employed",
        3: "Living with two parents, only Mother is employed",
        4: "Living with two parents, neither employed",
        5: "Living with Father, and Father is employed",
        6: "Living with Father, and Father is not employed",
        7: "Living with Mother, and Mother is employed",
        8: "Living with Mother, and Mother is not employed",
    },
    missing_value_fill="N/A (not own child of householder, and not child in subfamily)",
    verbalize=lambda x: (
        f"The person is {x.lower}"
        if not x.startswith("N/A")
        else "The person is not the own child of the householder and not a child in a subfamily."
    ),
)

# CIT: Citizenship Status
acs_citizenship = ColumnToText(
    "CIT",
    short_description="citizenship status",
    value_map={
        1: "Born in the United States",
        2: "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas",
        3: "Born abroad of American parents",
        4: "Naturalized US citizen",
        5: "Not a US citizen",
    },
    verbalize=lambda x: f"The person's citizenship status is '{x}'.",
)

# MIG: Mobility Status
acs_mobility = ColumnToText(
    "MIG",
    short_description="mobility status over the last year",
    value_map={
        1: "Lived in the same house one year ago",
        2: "Lived in a different house, outside the United States and Puerto Rico, one year ago",
        3: "Lived in a different house in the United States one year ago",
    },
    verbalize=lambda x: f"Considering the mobility status over the last year, the person {x[0].lower() + x[1:]}.",
)

# MIG: Mobility Status (Thresholded)
acs_mobility_qa = MultipleChoiceQA(
    column=acs_mobility_threshold.apply_to_column_name(
        "MIG"
    ),  # NOTE: Thresholded by MIG!=1
    text="Has this person moved in the last year?",
    choices=(
        Choice("No, person has lived in the same house for the last year", 0),
        Choice("Yes, person has moved in the last year", 1),
    ),
)

acs_mobility_numeric_qa = DirectNumericQA(
    column=acs_mobility_threshold.apply_to_column_name(
        "MIG"
    ),  # NOTE: Thresholded by MIG!=1
    text=(
        "What is the probability that this person has moved in the last year?"
    ),  # NOTE: Question should relate to probability of MIG!=1
)

acs_mobility_target_col = ColumnToText(
    name=acs_mobility_threshold.apply_to_column_name("MIG"),
    short_description="mobility status over the last year",
    question=acs_mobility_qa,
    use_value_map_only=True,
    verbalize=lambda x: f"Considering the mobility status over the last year, the person {x[0].lower() + x[1:]}.",
)

# MIL: Military Service Status
acs_military = ColumnToText(
    "MIL",
    short_description="military service status",
    value_map={
        1: "Now on active duty",
        2: "On active duty in the past, but not now",
        3: "Only on active duty for training in Reserved/National Guard",
        4: "Never served in the military",
    },
    missing_value_fill="N/A (less than 17 years old)",
    verbalize=lambda x: (
        f"The person {'is' if x.startswith('Now') else 'has'}{' been' if x.startswith('On') else ''}{x[0].lower()+x[1:]}."
        if not x.startswith("N/A")
        else "The person has not served the military, because the person is less than 17 years old."
    ),
)

# ANC: Ancestry
acs_ancestry = ColumnToText(
    "ANC",
    short_description="ancestry",
    value_map={
        1: "Single ancestry",
        2: "Multiple ancestry",
        3: "Unclassified",
        4: "Not reported",
        8: "N/A (information suppressed for certain area codes)",
    },
    verbalize=lambda x: f"The person's ancenstry is {x.lower() if not x.startswith('N/A') else 'not available'}.",
)

# ANC1P: Detailed Ancestry
acs_detailed_ancestry = ColumnToText(
    "ANC1P",
    short_description="detailed ancestry",
    value_map=partial(
        parse_pums_code,
        file=ACS_CODEBOOK_DIR / "ANC1P.txt",
        postprocess=lambda x: x.strip(),
    ),
    verbalize=lambda x: f"The person's ancenstry is {x.lower() if not x.startswith('N/A') else 'not available'}.",
)

# NATIVITY: Nativity
acs_nativity = ColumnToText(
    "NATIVITY",
    short_description="nativity",
    value_map={
        1: "Native",
        2: "Foreign born",
    },
    verbalize=lambda x: f"The person's nativity is {x.lower()}.",
)

# LANX: Language other than English spoken at home
acs_lanx = ColumnToText(
    "LANX",
    short_description="language other than English spoken at home",
    value_map={
        1: "Speaks a language other than English at home.",
        2: "Speaks only English at home.",
    },
    use_value_map_only=True,
    missing_value_fill="N/A (less than 5 years old)",
    verbalize=lambda x: f"The person {x[0].lower() + x[1:] if not x.startswith('N/A') else 'only English'}.",
)

# LANP: Language spoken at home
acs_lanp = ColumnToText(
    "LANP",
    short_description="language spoken at home (other than English)",
    value_map=partial(
        parse_pums_code,
        file=ACS_CODEBOOK_DIR / "LANP.txt",
        postprocess=lambda x: x.strip(),
    ),
    missing_value_fill="N/A (speaks only English at home)",
    verbalize=lambda x: f"The person' speaks {x if not x.startswith('N/A') else 'only English'} at home.",
)

# ENG: Ability to Speak English
acs_eng = ColumnToText(
    "ENG",
    short_description="ability to speak English",
    value_map={
        1: "Very well",
        2: "Well",
        3: "Not well",
        4: "Not at all",
    },
    missing_value_fill="Native (speaks only English)",
    verbalize=lambda x: f"The person speaks English {x.lower() if not x.startswith('N/A') else 'natively'}.",
)

# NOP: Nativity of Parents
acs_nop = ColumnToText(
    "NOP",
    short_description="nativity of parents",
    value_map={
        1: "Living with both parents, both of whom are native",
        2: "Living with both parents, Father only is foreign born",
        3: "Living with both parents, Mother only is foreign born",
        4: "Living with both parents, both of whom are foreign born",
        5: "Living with Father only, who is native",
        6: "Living with Father only, who is foreign born",
        7: "Living with Mother only, who is native",
        8: "Living with Mother only, who is foreign born",
    },
    missing_value_fill=(
        "N/A (not own child of householder and not child in subfamily)"
        # OR greater than 17 years old
    ),
    verbalize=lambda x: f"The person is {x.lower() if not x.startswith('N/A') else 'not own child of householder and not child in subfamily or greater tahn 17 years old.'}.",
)

# DEAR: Hearing Status
acs_hearing = ColumnToText(
    "DEAR",
    short_description="hearing status",
    value_map={
        1: "With hearing difficulty",
        2: "No hearing difficulty",
    },
    verbalize=lambda x: f"The person has {'a' if x.startswith('With') else 'no'} hearing difficulty.",
)

# DEYE: Vision Status
acs_vision = ColumnToText(
    "DEYE",
    short_description="vision status",
    value_map={
        1: "With vision difficulty",
        2: "No vision difficulty",
    },
    verbalize=lambda x: f"The person has {'a' if x.startswith('With') else 'no'} vision difficulty.",
)

# DREM: Cognitive Status
acs_cognitive = ColumnToText(
    "DREM",
    short_description="cognition status",
    value_map={
        1: "With cognitive difficulty",
        2: "No cognitive difficulty",
    },
    missing_value_fill="N/A (less than 5 years old)",
    verbalize=lambda x: (
        f"The person has {'a' if x.startswith('With') else 'no'} cognitive difficulty."
        if not x.startswith("N/A")
        else "No information on cognitive difficulties is availbale, because the person is less than 5 years old"
    ),
)

# ESR: Employment Status
acs_employment = ColumnToText(
    "ESR",
    short_description="employment status",
    value_map={
        1: "Civilian employed, at work",
        2: "Civilian employed, with a job but not at work",
        3: "Unemployed",
        4: "Armed forces, at work",
        5: "Armed forces, with a job but not at work",
        6: "Not in labor force",
    },
    missing_value_fill="N/A (less than 16 years old)",
    verbalize=lambda x: (
        f"The person's employment status is '{x}'."
        if not x.startswith("N/A")
        else "No employment status is provided, because the person is less than 16 years old."
    ),
)

# ESR: Employment Status (Thresholded)
acs_employment_qa = MultipleChoiceQA(
    column=acs_employment_threshold.apply_to_column_name("ESR"),
    text="What is this person's employment status?",
    choices=(
        Choice("Employed civilian", 1),
        Choice("Unemployed or in the military", 0),
    ),
)

acs_employment_numeric_qa = DirectNumericQA(
    column=acs_employment_threshold.apply_to_column_name("ESR"),
    text=(
        "What is the probability that this person is an employed civilian?"
        # "What is the probability that this person is currently employed?"
    ),  # NOTE: Question should relate to probability of ESR==1
)

acs_employment_target_col = ColumnToText(
    name=acs_employment_threshold.apply_to_column_name("ESR"),
    short_description="employment status",
    question=acs_employment_qa,
    use_value_map_only=True,
    verbalize=lambda x: (
        f"The person's employment status is '{x}'."
        if not x.startswith("N/A")
        else "No employment status is provided, because the person is less than 16 years old."
    ),
)

# ST: State
acs_state = ColumnToText(
    "ST",
    short_description="resident state",
    value_map=partial(
        parse_pums_code,
        file=ACS_CODEBOOK_DIR / "ST.txt",
        postprocess=lambda x: x[: x.find("/")].strip(),
    ),
    verbalize=lambda x: f"The person lives in {x}.",
)

# HISP: Hispanic Origin (Detailed)
acs_hispanic = ColumnToText(
    "HISP",
    short_description="Hispanic origin",
    value_map=partial(
        parse_pums_code,
        file=ACS_CODEBOOK_DIR / "HISP.txt",
        postprocess=lambda x: x.strip(),
    ),
    verbalize=lambda x: f"The person is of {x} origin.",
)

# FER: Parenthood Status
acs_parenthood = ColumnToText(
    "FER",
    short_description="person has given birth within the last year",
    use_value_map_only=True,
    value_map={
        1: "Person has given birth within the last year.",
        2: "Person has not given birth within the last year.",
    },
    missing_value_fill="N/A (less than 15 years old, or greater than 50 years old, or male)",
    verbalize=lambda x: (
        f"The person {x[7:]}"
        if not x.startswith("N/A")
        else "No parenthood status provided, because the person is less than 15 years old, greater than 50 years old or male."
    ),
)

# JWMNP: Commute Time
acs_commute_time = ColumnToText(
    "JWMNP",
    short_description="commute time",
    value_map=lambda x: f"{int(x)} minutes",
    missing_value_fill="N/A (not a worker, or worker who worked at home)",
    verbalize=lambda x: (
        f"The person's commute time is {x}."
        if not x.startswith("N/A")
        else "No commute time is providd, because the person is not a worker or is a worker who worked at home."
    ),
)

# JWMNP: Commute Time (Thresholded)
acs_commute_time_qa = MultipleChoiceQA(
    column=acs_travel_time_threshold.apply_to_column_name("JWMNP"),
    text="What is this person's commute time?",
    choices=(
        Choice("Longer than 20 minutes", 1),
        Choice("Less than 20 minutes", 0),
    ),
)

acs_commute_time_numeric_qa = DirectNumericQA(
    column=acs_travel_time_threshold.apply_to_column_name("JWMNP"),
    text=(
        "What is the probability that this person's commute time is longer than 20 minutes?"
    ),  # NOTE: Question should relate to probability of JWMNP>20
)

acs_travel_time_target_col = ColumnToText(
    name=acs_travel_time_threshold.apply_to_column_name("JWMNP"),
    short_description="commute time",
    question=acs_commute_time_qa,
    use_value_map_only=True,
    verbalize=lambda x: (
        f"The person's commute time is {x}."
        if not x.startswith("N/A")
        else "No commute time is provided, because the person is not a worker or is a worker who worked at home."
    ),
)

# JWTR: Commute Method
acs_commute_method = ColumnToText(
    "JWTR",
    short_description="means of transportation to work",
    value_map={
        1: "Car, truck, or van",
        2: "Bus or trolley bus",
        3: "Streetcar or trolley car",
        4: "Subway or elevated",
        5: "Railroad",
        6: "Ferryboat",
        7: "Taxicab",
        8: "Motorcycle",
        9: "Bicycle",
        10: "Walked",
        11: "Worked at home",
        12: "Other method",
    },
    verbalize=lambda x: (
        f"The person's means of transportation to work is a {x.lower()}."
        if not (
            x.startswith("Other") or x.startswith("Walked") or x.startswith("Worked")
        )
        else (
            "The person uses other means of transportation to work."
            if x.startswith("Other")
            else f"To get to work, the person {x.lower()}."
        )
    ),
)


def describe_income_to_poverty_ratio_pct(ratio_pct: float) -> str:
    """Describe the income-to-poverty ratio in natural language."""
    context_msg: str

    if ratio_pct < 90:
        context_msg = "below the poverty line"
    elif 90 <= ratio_pct < 110:
        context_msg = "at or near the poverty line"
    else:
        context_msg = "above the poverty line"

    return f"{int(ratio_pct)}% of the poverty line income, which is {context_msg}"


# POVPIP: Income-to-Poverty Ratio
acs_poverty_ratio = ColumnToText(
    "POVPIP",
    short_description="income-to-poverty ratio",
    value_map=describe_income_to_poverty_ratio_pct,
    verbalize=lambda x: f"The person's income corresponds to {x}.",
)

# POVPIP: Income-to-Poverty Ratio (Thresholded)
acs_poverty_ratio_qa = MultipleChoiceQA(
    column=acs_poverty_ratio_threshold.apply_to_column_name("POVPIP"),
    text=(
        "Is this person's income-to-poverty ratio below 2.5 ? "
        "That is, is this person's annual income below 2.5 times the poverty line income?",
    ),
    choices=(
        Choice("Yes, this person earns below 2.5 times the poverty line income", 1),
        Choice("No, this person earns above 2.5 times the poverty line income.", 0),
    ),
)

acs_poverty_ratio_target_col = ColumnToText(
    name=acs_poverty_ratio_threshold.apply_to_column_name("POVPIP"),
    short_description="income-to-poverty ratio is below 2.5",
    question=acs_poverty_ratio_qa,
    use_value_map_only=True,
    verbalize=lambda x: f"The person's income corresponds to {x}.",
)

# GCL: Grandparent Living with Grandchildren
acs_gcl_col = ColumnToText(
    "GCL",
    short_description="grandparent living with grandchildren",
    use_value_map_only=True,
    value_map={
        1: "Household includes grandparent living with grandchildren",
        2: "Household does not include grandparents living with grandchildren",
    },
    missing_value_fill="N/A (less than 30 years old, or living in institutional group quarters)",
    verbalize=lambda x: (
        x
        if not x.startswith("N/A")
        else "The person is less than 30 years old or living in institutional group quarters."
    ),
)

# PUMA: Public Use Microdata Area Code
# TODO: assign meaningful natural-text values to PUMA codes
# > Consult: https://www2.census.gov/geo/pdfs/reference/puma/2010_PUMA_Names.pdf
acs_puma_col = ColumnToText(
    "PUMA",
    short_description="Public Use Microdata Area (PUMA) code",
    use_value_map_only=True,
    value_map=lambda x: f"Public Use Microdata Area (PUMA) code: {int(x)}.",
    # missing_value_fill="N/A (less than 16 years old)",
    verbalize=lambda x: f"The person's {x}.".replace(":", " is"),
)

# POWPUMA: Place of Work PUMA
acs_powpuma_col = ColumnToText(
    "POWPUMA",
    short_description="place of work PUMA",
    use_value_map_only=True,
    value_map=lambda x: f"Public Use Microdata Area (PUMA) code for the place of work: {int(x)}.",
    # missing_value_fill="N/A (not a worker, or worker who worked at home)",
    verbalize=lambda x: f"The person's {x}.".replace(":", " is"),
)

# HINS2: Health Insurance Coverage through Private Company
acs_health_ins_2_col = ColumnToText(
    "HINS2",
    short_description="acquired health insurance directly from an insurance company",
    use_value_map_only=True,
    value_map={
        1: "Person has purchased insurance directly from an insurance company",
        2: (
            "Person has not purchased insurance directly from an insurance "
            "company (is either uninsured or insured through another source)",
        ),
    },
    verbalize=lambda x: f"The {x[0].lower()+x[1:]}.",
)

# HINS2: Health Insurance Coverage through Private Company (Thresholded)
acs_health_ins_2_qa = MultipleChoiceQA(
    column=acs_health_insurance_threshold.apply_to_column_name("HINS2"),
    text="Has this person purchased health insurance directly from an insurance company?",
    choices=(
        Choice("Yes, this person has health insurance through a private company", 1),
        Choice(
            "No, this person either has insurance through other means or is uninsured",
            0,
        ),
    ),
)

acs_health_ins_2_numeric_qa = DirectNumericQA(
    column=acs_health_insurance_threshold.apply_to_column_name("HINS2"),
    text=(
        "What is the probability that this person has purchased health "
        "insurance directly through a private company?"
    ),  # NOTE: Question should relate to probability of HINS2==1
)

acs_health_ins_2_target_col = ColumnToText(
    name=acs_health_insurance_threshold.apply_to_column_name("HINS2"),
    short_description="acquired health insurance directly from an insurance company",
    question=acs_health_ins_2_qa,
    use_value_map_only=True,
    verbalize=lambda x: f"The {x[0].lower()+x[1:]}.",
)
