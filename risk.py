"""risk.py - turn a probability into a risk level plus warning flags."""


def risk_level(prob, credit_history=None, debt_to_income=None):
    if prob >= 0.75:
        label = "Low Risk"
    elif prob >= 0.45:
        label = "Medium Risk"
    else:
        label = "High Risk"

    flags = []
    if credit_history == 0:
        flags.append("Poor or missing credit history")
    if debt_to_income is not None and debt_to_income > 0.40:
        flags.append("EMI is more than 40% of income")

    if flags and label == "Low Risk":
        label = "Medium Risk"
    return label, flags