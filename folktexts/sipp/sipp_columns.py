"""
SIPP task columns

Value Maps adapted from https://github.com/mlfoundations/tableshift/blob/main/tableshift/datasets/brfss.py

For more information on the dataset and raw data see:
* https://www2.census.gov/programs-surveys/sipp/data/datasets/2014/w1/
* https://www2.census.gov/programs-surveys/sipp/data/datasets/2014/w2/

"""

# from functools import partial
# from pathlib import Path

from folktexts.threshold import Threshold

from ..col_to_text import ColumnToText
from ..qa_interface import Choice, DirectNumericQA, MultipleChoiceQA

# SIPP Threshold
sipp_threshold = Threshold(1, "==")

# SIPP Columns
sipp_livqrt = ColumnToText(
    name="LIVING_QUARTERS_TYPE",
    short_description="type of living quarters",
    value_map={
        1.0: "House, apartment, flat",
        2.0: "Other",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_livown = ColumnToText(
    name="LIVING_OWNERSHIP",
    short_description="ownership status of living quarters",
    value_map={
        1.0: "Owned or being bought by someone in the household",
        2.0: "Rented",
        3.0: "Occupied without payment of rent",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_snap = ColumnToText(
    name="SNAP_ASSISTANCE",
    short_description="percentage of the year the respondent received SNAP/food stamps assistance",
    value_map=lambda x: f"{x * 100:.2f}%",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_wic = ColumnToText(
    name="WIC_ASSISTANCE",
    short_description="percentage of the year the respondent received WIC assistance",
    value_map=lambda x: f"{x * 100:.2f}%",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_medicare = ColumnToText(
    name="MEDICARE_ASSISTANCE",
    short_description="percentage of the year the respondent was covered by Medicare",
    value_map=lambda x: f"{x * 100:.2f}%",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_medicaid = ColumnToText(
    name="MEDICAID_ASSISTANCE",
    short_description="percentage of the year the respondent was covered by Medicaid",
    value_map=lambda x: f"{x * 100:.2f}%",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_healthdisab = ColumnToText(
    name="HEALTHDISAB",
    short_description=(
        "has a physical, mental or other health condition that limits the kind or amount of work he/she can do"
    ),
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_dayssick = ColumnToText(
    name="DAYS_SICK",
    short_description="number of days sick in last year",
    value_map=lambda x: f"{int(x)} days",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_hospnights = ColumnToText(
    name="HOSPITAL_NIGHTS",
    short_description="number of nights in hospital last year",
    value_map=lambda x: f"{int(x)} nights",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_prescriptions = ColumnToText(
    name="PRESCRIPTION_MEDS",
    short_description="uses prescription medications",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_dentist = ColumnToText(
    name="VISIT_DENTIST_NUM",
    short_description="number of dentist visits",
    value_map=lambda x: f"{int(x)} vists",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_doctor = ColumnToText(
    name="VISIT_DOCTOR_NUM",
    short_description="number of visits to a doctor, nurse, or any other type of medical provider",
    value_map=lambda x: f"{int(x)} vists",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_ins_premiums = ColumnToText(
    name="HEALTH_INSURANCE_PREMIUMS",
    short_description="amount paid for comprehensive health insurance premiums",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_otc_pay = ColumnToText(
    name="HEALTH_OVER_THE_COUNTER_PRODUCTS_PAY",
    short_description="amount paid for over-the-counter health-related products",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_med_care_pay = ColumnToText(
    name="HEALTH_MEDICAL_CARE_PAY",
    short_description="amount paid for non-premium medical out-of-pocket expenditures on medical care",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_hearing = ColumnToText(
    name="HEALTH_HEARING",
    short_description="is deaf of has hearing difficulties",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_seeing = ColumnToText(
    name="HEALTH_SEEING",
    short_description="is blind or has vision difficulties",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_cognitive = ColumnToText(
    name="HEALTH_COGNITIVE",
    short_description="has serious difficulty concentrating, remembering, or making decisions",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_ambulatory = ColumnToText(
    name="HEALTH_AMBULATORY",
    short_description="has serious difficulty walking or climbing stairs",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_selfcare = ColumnToText(
    name="HEALTH_SELF_CARE",
    short_description="has difficulty with self-care such as dressing or bathing",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_errands = ColumnToText(
    name="HEALTH_ERRANDS_DIFFICULTY",
    short_description="has difficulty doing errands alone",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_core_disab = ColumnToText(
    name="HEALTH_CORE_DISABILITY",
    short_description="has a core disability",
    value_map={
        1.0: "Yes, with a core disability",
        2.0: "No, without a core disability",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_supp_disab = ColumnToText(
    name="HEALTH_SUPPLEMENTAL_DISABILITY",
    short_description=(
        "answered positively to at least one core questions, three child disability questions, "
        "or two work disability questions"
    ),
    value_map={
        1.0: "Yes, with a disability",
        2.0: "No, without a disability",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_age = ColumnToText(
    name="AGE",
    short_description="age",
    value_map=lambda x: f"{int(x)} years old",
    missing_value_fill="N/A (refused or unknown)",
)
# Min: 0 Max: 90

sipp_gender = ColumnToText(
    name="GENDER",
    short_description="gender of person",
    value_map={
        1.0: "Male",
        2.0: "Female",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_race = ColumnToText(
    name="RACE",
    short_description="races the person identifies with",
    value_map={
        1.0: "White only",
        2.0: "Black only",
        3.0: "American Indian or Alaska Native only",
        4.0: "Asian only",
        5.0: "Native Hawaiian or Other Pacific Islander only",
        6.0: "White and Black",
        7.0: "White and American Indian or Alaska Native",
        8.0: "White and Asian",
        9.0: "White and Native Hawaiian or Other Pacific Islander",
        10.0: "Black and American Indian or Alaska Native",
        11.0: "Black and Asian",
        12.0: "Black and Native Hawaiian or Other Pacific Islander",
        13.0: "American Indian or Alaska Native and Asian",
        14.0: "Asian and Native Hawaiian or Other Pacific Islander",
        15.0: "White, Black and American Indian or Alaska Native",
        16.0: "White, Black and Asian",
        17.0: "White, American Indian or Alaska Native and Asian",
        18.0: "White, Asian and Native Hawaiian or Other Pacific Islander",
        19.0: "White, Black, American Indian or Alaska Native and Asian",
        20.0: "Other 2 or 3 races",
        21.0: "Other 4 or 5 races",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_education = ColumnToText(
    name="EDUCATION",
    short_description="highest level of education completed",
    value_map={
        31.0: "Less than 1st grade",
        32.0: "1st, 2nd, 3rd or 4th grade",
        33.0: "5th or 6th grade",
        34.0: "7th or 8th grade",
        35.0: "9th grade",
        36.0: "10th grade",
        37.0: "11th grade",
        38.0: "12th grade, no diploma",
        39.0: "High School Graduate (diploma or GED or equivalent)",
        40.0: "Some college credit, but less than 1 year (regular Junior college/college/university)",
        41.0: "1 or more years of college, no degree (regular Junior college/college/university)",
        42.0: "Associate's degree (2-year college)",
        43.0: "Bachelor's degree (for example: BA, AB, BS)",
        44.0: "Master's degree (for example: MA, MS, MBA, MSW)",
        45.0: "Professional School degree (for example: MD (doctor), DDS (dentist), JD (lawyer))",
        46.0: "Doctorate degree (for example: Ph.D., Ed.D.)",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_marital = ColumnToText(
    name="MARITAL_STATUS",
    short_description="marital status",
    value_map={
        1.0: "Married, spouse present",
        2.0: "Married, spouse absent",
        3.0: "Widowed",
        4.0: "Divorced",
        5.0: "Separated",
        6.0: "Never married",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_citizen = ColumnToText(
    name="CITIZENSHIP_STATUS",
    short_description="US citizenship status",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_famsize = ColumnToText(
    name="FAMILY_SIZE_AVG",
    short_description="average number of persons in family",
    value_map=lambda x: f"{int(x)} persons",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_origin = ColumnToText(
    name="ORIGIN",
    short_description="Spanish, Hispanic, or Latino origin",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_incomehh = ColumnToText(
    name="HOUSEHOLD_INC",
    short_description="total monthly income of all household members",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_workcomp = ColumnToText(
    name="RECEIVED_WORK_COMP",
    short_description="received worker's compensation payments",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_tanf = ColumnToText(
    name="TANF_ASSISTANCE",
    short_description="percentage of the year the respondent received TANF benefit",
    value_map=lambda x: f"{x * 100:.2f}%",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_unemp = ColumnToText(
    name="UNEMPLOYMENT_COMP",
    short_description="receives unemployment compensation payments",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_sevpay = ColumnToText(
    name="SEVERANCE_PAY_PENSION",
    short_description="receives any severance pay or lump sum payments from a pension or retirement plan",
    value_map={
        1.0: "Yes",
        2.0: "No",
    },
    missing_value_fill="N/A (refused or unknown)",
)

sipp_fostercare = ColumnToText(
    name="FOSTER_CHILD_CARE_AMT",
    short_description="amount of foster child care payments received per month",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_childsupport = ColumnToText(
    name="CHILD_SUPPORT_AMT",
    short_description="amount of child support payments received per month",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_alimony = ColumnToText(
    name="ALIMONY_AMT",
    short_description="amount of alimony payments received per month",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_inc_assist = ColumnToText(
    name="INCOME_FROM_ASSISTANCE",
    short_description="total income from public assistance, benefits or compensation",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_income = ColumnToText(
    name="INCOME",
    short_description="total personal income",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_savings = ColumnToText(
    name="SAVINGS_INV_AMOUNT",
    short_description="total value of retirement accounts",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_unemp_amt = ColumnToText(
    name="UNEMPLOYMENT_COMP_AMOUNT",
    short_description="amount of unemployment compensation per month",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_va = ColumnToText(
    name="VA_BENEFITS_AMOUNT",
    short_description="total amount of VA benefits per month",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_retire = ColumnToText(
    name="RETIREMENT_INCOME_AMOUNT",
    short_description="total amount of retirement income per month",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_survivor = ColumnToText(
    name="SURVIVOR_INCOME_AMOUNT",
    short_description="total amount of survivor benefits per month",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_disab_inc = ColumnToText(
    name="DISABILITY_BENEFITS_AMOUNT",
    short_description="total amount of disability benefits or income per month",
    value_map=lambda x: f"${int(x)}",
    missing_value_fill="N/A (refused or unknown)",
)

sipp_foodassist = ColumnToText(
    name="FOOD_ASSISTANCE",
    short_description="received food assistance",
    value_map={0.0: "No", 1.0: "Yes"},
    missing_value_fill="N/A (refused or unknown)",
)

sipp_transassist = ColumnToText(
    name="TRANSPORTATION_ASSISTANCE",
    short_description="received transportation assistance",
    value_map={0.0: "No", 1.0: "Yes"},
    missing_value_fill="N/A (refused or unknown)",
)

sipp_socialsec = ColumnToText(
    name="SOCIAL_SEC_BENEFITS",
    short_description="received Social Security benefits",
    value_map={0.0: "No", 1.0: "Yes"},
    missing_value_fill="N/A (refused or unknown)",
)

sipp_opm = ColumnToText(
    name="OPM_RATIO",
    short_description="Household income-to-poverty ratio in this calendar year is equal or larger than 3",
    value_map={0.0: "No", 1.0: "Yes"},
    missing_value_fill="N/A (refused or unknown)",
)

sipp_qa = MultipleChoiceQA(
    column=sipp_threshold.apply_to_column_name("OPM_RATIO"),
    text=(
        "Is this household's income at least three times the poverty threshold"
        "based on the Officual Poverty Measure (OPM)?"
    ),
    choices=(
        Choice("Yes, this household's income is at least three times the poverty threshold", 1),
        Choice("No, this household's income is at less than three times the poverty threshold", 0),
    ),
)

sipp_numeric_qa = DirectNumericQA(
    column=sipp_threshold.apply_to_column_name("OPM_RATIO"),
    text=(
        "What is the probability that this individual's income is "
        "at least three times the poverty threshold based on the Official Poverty Measure (OPM)?"
    ),
)

# We consider Wave 1 and Wave 2 of the SIPP 2014 panel data.
# The target variable is based on the official poverty measure (OPM), a cash-income based measure of poverty.
sipp_target_col = ColumnToText(
    name=sipp_threshold.apply_to_column_name("OPM_RATIO"),
    short_description=(
        "Is this individual's income at least three times the poverty threshold "
        "based on the Official Poverty Measure (OPM)"
    ),
    value_map={
        0: "No",
        1: "Yes",
    },
    missing_value_fill="N/A (refused or unknown)",
    question=sipp_qa,
)
