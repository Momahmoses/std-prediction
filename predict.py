"""
STD Risk Prediction — Interactive CLI Tool
Run: python3 predict.py

NOTE: This tool is for educational/public health awareness purposes only.
      It is NOT a substitute for professional medical testing and advice.
"""

import joblib
import numpy as np

model    = joblib.load('models/best_model.pkl')
scaler   = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/encoders.pkl')
metadata = joblib.load('models/metadata.pkl')

THRESHOLD = 0.40

def get_input(prompt, type_fn, valid=None, min_val=None, max_val=None):
    while True:
        try:
            val = type_fn(input(prompt).strip())
            if valid is not None and val not in valid:
                print(f"  Please enter one of: {valid}")
                continue
            if min_val is not None and val < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return val
        except (ValueError, KeyError):
            print("  Invalid input, please try again.")

def predict():
    print("\n" + "="*60)
    print("     STD RISK ASSESSMENT & PREDICTION TOOL")
    print("="*60)
    print("  All responses are confidential. Answer honestly")
    print("  for the most accurate risk assessment.\n")

    age           = get_input("Your age: ", int, min_val=13, max_val=100)
    gender        = get_input("Gender (male/female): ", str, valid=['male','female'])
    num_partners  = get_input("Number of sexual partners (last 12 months): ", int, min_val=0, max_val=50)
    condom_use    = get_input("Condom use (always/sometimes/never): ", str,
                               valid=['always','sometimes','never'])
    prev_std      = get_input("Have you had an STD before? (1=Yes, 0=No): ", int, valid=[0,1])
    iv_drug       = get_input("Do you use intravenous drugs? (1=Yes, 0=No): ", int, valid=[0,1])
    alcohol       = get_input("Alcohol use (none/moderate/heavy): ", str,
                               valid=['none','moderate','heavy'])
    education     = get_input("Education level (none/primary/secondary/tertiary): ", str,
                               valid=['none','primary','secondary','tertiary'])
    hiv           = get_input("Are you HIV positive? (1=Yes, 0=No): ", int, valid=[0,1])
    test_freq     = get_input("How often do you get STI tested?\n"
                               "  Options: never / yearly / every_6_months / every_3_months\n"
                               "  Your answer: ", str,
                               valid=['never','yearly','every_6_months','every_3_months'])
    area          = get_input("Do you live in an (urban/rural) area? ", str, valid=['urban','rural'])
    rel_status    = get_input("Relationship status (single/married/cohabiting/casual): ", str,
                               valid=['single','married','cohabiting','casual'])

    print("\nDo you currently have any of these symptoms? (1=Yes, 0=No)")
    discharge = get_input("  Unusual discharge: ", int, valid=[0,1])
    sores     = get_input("  Genital sores/ulcers: ", int, valid=[0,1])
    burning   = get_input("  Burning sensation during urination: ", int, valid=[0,1])
    rash      = get_input("  Unexplained rash: ", int, valid=[0,1])

    age_first = get_input("\nAge at first sexual intercourse: ", int, min_val=10, max_val=50)

    # ── Encode ─────────────────────────────────────────────────────────────────
    enc = encoders
    features = np.array([[
        age,
        enc['gender'].transform([gender])[0],
        num_partners,
        enc['condom_use'].transform([condom_use])[0],
        prev_std,
        iv_drug,
        enc['alcohol_use'].transform([alcohol])[0],
        enc['education'].transform([education])[0],
        hiv,
        enc['sti_test_freq'].transform([test_freq])[0],
        enc['urban_rural'].transform([area])[0],
        enc['relationship_status'].transform([rel_status])[0],
        discharge, sores, burning, rash,
        age_first
    ]])

    if metadata['scaled']:
        features = scaler.transform(features)

    prob      = model.predict_proba(features)[0][1]
    predicted = int(prob >= THRESHOLD)

    if prob < 0.25:
        risk_level = "LOW"
        color      = "SAFE"
    elif prob < 0.45:
        risk_level = "MODERATE"
        color      = "CAUTION"
    elif prob < 0.70:
        risk_level = "HIGH"
        color      = "WARNING"
    else:
        risk_level = "VERY HIGH"
        color      = "URGENT"

    print("\n" + "="*60)
    print("  STD RISK ASSESSMENT RESULT")
    print("="*60)
    print(f"  STD Risk Probability : {prob*100:.1f}%")
    print(f"  Risk Level           : {risk_level} [{color}]")
    print(f"  Prediction           : {'LIKELY POSITIVE — Seek testing immediately' if predicted else 'LIKELY NEGATIVE — Maintain safe practices'}")
    print("="*60)

    print("\n  RECOMMENDATIONS:")
    if predicted:
        print("  - Visit a clinic or hospital for STD testing immediately")
        print("  - Do not engage in sexual activity until tested & treated")
        print("  - Inform recent partners so they can also get tested")
        if discharge or sores or burning or rash:
            print("  - Your symptoms require urgent medical attention")
        if hiv:
            print("  - Continue antiretroviral therapy as prescribed")
    else:
        print("  - Continue practicing safe sex")
        if condom_use != 'always':
            print("  - Use condoms consistently to reduce risk further")
        if test_freq == 'never':
            print("  - Consider getting tested at least once a year")
        if num_partners > 3:
            print("  - Reducing number of partners lowers your risk")

    print("\n  DISCLAIMER: This tool is for awareness only.")
    print("  Always consult a qualified healthcare professional.")
    print("="*60)

    again = input("\nAssess another person? (y/n): ").strip().lower()
    if again == 'y':
        predict()

if __name__ == '__main__':
    predict()
