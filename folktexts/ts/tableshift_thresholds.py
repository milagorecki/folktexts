"""Threshold instances for ACS / folktables tasks."""

from folktexts.threshold import Threshold

# BRFSS Diabetes Task
brfss_diabetes_threshold = Threshold(1, "==")

# BRFSS Hypertension Task
brfss_hypertension_threshold = Threshold(1, "==")
