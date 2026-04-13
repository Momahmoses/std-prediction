"""
Generates a synthetic STD (Sexually Transmitted Disease) risk dataset.
Features are based on real-world epidemiological risk factors.
Target: std_positive (1 = tested positive, 0 = negative)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 8000

age                  = np.random.randint(15, 65, N)
gender               = np.random.choice(['male', 'female'], N, p=[0.48, 0.52])
num_partners         = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], N,
                           p=[0.30,0.20,0.15,0.10,0.08,0.06,0.04,0.03,0.02,0.02])
condom_use           = np.random.choice(['always','sometimes','never'], N, p=[0.35,0.40,0.25])
prev_std             = np.random.binomial(1, 0.20, N)
iv_drug_use          = np.random.binomial(1, 0.05, N)
alcohol_use          = np.random.choice(['none','moderate','heavy'], N, p=[0.30,0.50,0.20])
education            = np.random.choice(['none','primary','secondary','tertiary'], N,
                           p=[0.10,0.25,0.40,0.25])
hiv_positive         = np.random.binomial(1, 0.04, N)
sti_test_freq        = np.random.choice(['never','yearly','every_6_months','every_3_months'],
                           N, p=[0.40,0.30,0.20,0.10])
urban_rural          = np.random.choice(['urban','rural'], N, p=[0.55,0.45])
relationship_status  = np.random.choice(['single','married','cohabiting','casual'], N,
                           p=[0.35,0.30,0.20,0.15])
symptom_discharge    = np.random.binomial(1, 0.15, N)
symptom_sores        = np.random.binomial(1, 0.10, N)
symptom_burning      = np.random.binomial(1, 0.18, N)
symptom_rash         = np.random.binomial(1, 0.08, N)
age_first_sex        = np.random.randint(13, 25, N)

# ── Risk score ─────────────────────────────────────────────────────────────────
risk = (
      0.05  * (num_partners / 10)
    + 0.20  * (condom_use == 'never').astype(int)
    + 0.10  * (condom_use == 'sometimes').astype(int)
    + 0.15  * prev_std
    + 0.18  * iv_drug_use
    + 0.08  * (alcohol_use == 'heavy').astype(int)
    + 0.04  * (alcohol_use == 'moderate').astype(int)
    + 0.12  * hiv_positive
    - 0.08  * (sti_test_freq == 'every_3_months').astype(int)
    - 0.05  * (sti_test_freq == 'every_6_months').astype(int)
    - 0.06  * (education == 'tertiary').astype(int)
    - 0.03  * (education == 'secondary').astype(int)
    + 0.04  * (urban_rural == 'urban').astype(int)
    + 0.06  * (relationship_status == 'casual').astype(int)
    + 0.10  * symptom_discharge
    + 0.10  * symptom_sores
    + 0.08  * symptom_burning
    + 0.06  * symptom_rash
    + 0.03  * (age_first_sex < 16).astype(int)
    + 0.02  * (age < 25).astype(int)
    + np.random.normal(0, 0.05, N)
).clip(0, 1)

std_positive = (risk > 0.35).astype(int)

df = pd.DataFrame({
    'age':                 age,
    'gender':              gender,
    'num_partners':        num_partners,
    'condom_use':          condom_use,
    'prev_std':            prev_std,
    'iv_drug_use':         iv_drug_use,
    'alcohol_use':         alcohol_use,
    'education':           education,
    'hiv_positive':        hiv_positive,
    'sti_test_freq':       sti_test_freq,
    'urban_rural':         urban_rural,
    'relationship_status': relationship_status,
    'symptom_discharge':   symptom_discharge,
    'symptom_sores':       symptom_sores,
    'symptom_burning':     symptom_burning,
    'symptom_rash':        symptom_rash,
    'age_first_sex':       age_first_sex,
    'std_positive':        std_positive
})

df.to_csv('data/std_dataset.csv', index=False)
print(f"Dataset created : {len(df):,} records")
print(f"STD Positive    : {std_positive.sum():,} ({std_positive.mean()*100:.1f}%)")
print(f"STD Negative    : {(std_positive==0).sum():,} ({(std_positive==0).mean()*100:.1f}%)")
print(df.head())
