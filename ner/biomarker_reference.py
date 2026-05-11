"""
Clinical reference ranges for common biomarkers.
Used by DiagnosticAgent to interpret values.
"""

BIOMARKER_RANGES = {
    "HbA1c": {
        "unit": "%", "normal_high": 5.6,
        "prediabetes_low": 5.7, "prediabetes_high": 6.4,
        "diabetes_threshold": 6.5,
        "trial_note": "Many diabetes trials require HbA1c > 7.5% or > 8%",
    },
    "Fasting Glucose": {
        "unit": "mg/dL", "normal_low": 70, "normal_high": 99,
        "prediabetes_low": 100, "prediabetes_high": 125,
        "diabetes_threshold": 126,
        "trial_note": "Diabetes trials often require fasting glucose > 126",
    },
    "eGFR": {
        "unit": "mL/min/1.73m2", "normal_low": 60,
        "ckd_stage3_low": 30, "ckd_stage3_high": 59,
        "ckd_stage4_low": 15, "ckd_stage4_high": 29,
        "trial_note": "CKD trials often require eGFR 30-60 for stage 3",
    },
    "LDL": {
        "unit": "mg/dL", "optimal": 100,
        "borderline_low": 130, "borderline_high": 159, "high": 160,
        "trial_note": "Cardiovascular trials often require LDL > 130",
    },
    "HDL": {
        "unit": "mg/dL", "low_male": 40, "low_female": 50,
        "trial_note": "Low HDL is a cardiovascular risk factor used in trial criteria",
    },
    "Total Cholesterol": {
        "unit": "mg/dL", "normal_high": 200,
        "borderline_high": 239, "high": 240,
    },
    "Creatinine": {
        "unit": "mg/dL",
        "normal_high_male": 1.2, "normal_high_female": 1.1,
    },
    "WBC": {
        "unit": "K/uL", "normal_low": 4.5, "normal_high": 11.0,
        "trial_note": "Oncology trials often require WBC > 3.0",
    },
    "Hemoglobin": {
        "unit": "g/dL",
        "normal_low_male": 13.5, "normal_high_male": 17.5,
        "normal_low_female": 12.0, "normal_high_female": 15.5,
        "anemia_threshold_male": 13.0, "anemia_threshold_female": 12.0,
        "trial_note": "Many trials exclude patients with hemoglobin < 10 g/dL",
    },
    "Platelets": {
        "unit": "K/uL", "normal_low": 150, "normal_high": 400,
        "trial_note": "Many trials require platelets > 100 K/uL",
    },
    "TSH": {
        "unit": "mIU/L", "normal_low": 0.4, "normal_high": 4.0,
        "hyperthyroid_threshold": 0.4, "hypothyroid_threshold": 4.0,
    },
    "ALT": {
        "unit": "U/L", "normal_high": 56,
        "trial_note": "Liver trials often exclude ALT > 3x upper limit",
    },
    "AST": {"unit": "U/L", "normal_high": 40},
    "BMI": {
        "unit": "kg/m2",
        "underweight": 18.5, "normal_high": 24.9,
        "overweight": 25.0, "obese": 30.0,
        "trial_note": "Obesity/metabolic trials often require BMI > 27 or > 30",
    },
    "Systolic BP": {
        "unit": "mmHg", "normal_high": 120,
        "elevated": 130, "high_stage1": 140,
        "trial_note": "Hypertension trials often require systolic BP > 140",
    },
    "Diastolic BP": {
        "unit": "mmHg", "normal_high": 80, "high_stage1": 90,
    },
}


def interpret_value(name: str, value: float, gender: str = None) -> str:
    """Return a plain English interpretation of a biomarker value."""
    ref = BIOMARKER_RANGES.get(name)
    if not ref:
        return f"{name}: {value} (no reference range available)"

    unit = ref.get("unit", "")

    if name == "HbA1c":
        if value >= 6.5:
            return (
                f"HbA1c of {value}{unit} is above the diagnostic threshold of 6.5% — "
                f"consistent with diabetes. Values above 7.5% are commonly required "
                f"for diabetes clinical trial enrollment."
            )
        elif value >= 5.7:
            return f"HbA1c of {value}{unit} falls in the prediabetes range (5.7–6.4%)."
        else:
            return f"HbA1c of {value}{unit} is within the normal range."

    if name == "eGFR":
        if value < 15:
            return f"eGFR of {value} {unit} indicates kidney failure (Stage 5 CKD)."
        elif value < 30:
            return f"eGFR of {value} {unit} indicates severe CKD (Stage 4)."
        elif value < 60:
            return (
                f"eGFR of {value} {unit} indicates moderate CKD (Stage 3) — "
                f"relevant for diabetic kidney disease trials."
            )
        else:
            return f"eGFR of {value} {unit} is within normal range."

    if name == "BMI":
        if value >= 30:
            return (
                f"BMI of {value} {unit} falls in the obese range. "
                f"Relevant for obesity and metabolic syndrome trials."
            )
        elif value >= 25:
            return f"BMI of {value} {unit} falls in the overweight range."
        else:
            return f"BMI of {value} {unit} is within normal range."

    normal_high = ref.get("normal_high")
    normal_low = ref.get("normal_low")
    trial_note = ref.get("trial_note", "")

    if normal_high and value > normal_high:
        msg = f"{name} of {value} {unit} is above the normal range (≤{normal_high})."
        if trial_note:
            msg += f" {trial_note}."
        return msg
    elif normal_low and value < normal_low:
        return f"{name} of {value} {unit} is below the normal range (≥{normal_low})."
    else:
        return f"{name} of {value} {unit} is within the normal range."
