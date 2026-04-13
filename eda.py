"""
Exploratory Data Analysis — STD Prediction Dataset
Generates visual insights saved to plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid')

df = pd.read_csv('data/std_dataset.csv')

print("="*55)
print("  STD PREDICTION — EXPLORATORY DATA ANALYSIS")
print("="*55)
print(f"\nShape        : {df.shape}")
print(f"\nClass split  :\n{df['std_positive'].value_counts().rename({0:'Negative',1:'Positive'})}")
print(f"\nMissing vals :\n{df.isnull().sum()}")
print(f"\nStats:\n{df.describe().round(2)}")

# ── Plot 1: Class Distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['std_positive'].value_counts()
bars = ax.bar(['STD Negative', 'STD Positive'], counts.values,
               color=['steelblue', 'crimson'], alpha=0.85)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontweight='bold')
ax.set_title('STD Test Result Distribution')
ax.set_ylabel('Number of Individuals')
plt.tight_layout()
plt.savefig('plots/01_class_distribution.png', dpi=150)
plt.close()

# ── Plot 2: STD Rate by Age Group ─────────────────────────────────────────────
df['age_group'] = pd.cut(df['age'], bins=[14,19,24,29,34,44,64],
                          labels=['15-19','20-24','25-29','30-34','35-44','45-64'])
fig, ax = plt.subplots(figsize=(9, 4))
rate = df.groupby('age_group', observed=True)['std_positive'].mean() * 100
bars = ax.bar(rate.index.astype(str), rate.values, color='darkorange', alpha=0.85)
for bar, val in zip(bars, rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontsize=9)
ax.set_title('STD Positive Rate by Age Group')
ax.set_xlabel('Age Group')
ax.set_ylabel('STD Positive Rate (%)')
plt.tight_layout()
plt.savefig('plots/02_std_by_age.png', dpi=150)
plt.close()

# ── Plot 3: STD Rate by Condom Use ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
rate = df.groupby('condom_use')['std_positive'].mean() * 100
order = ['always', 'sometimes', 'never']
rate = rate.reindex(order)
bars = ax.bar(order, rate.values, color=['green','orange','crimson'], alpha=0.85)
for bar, val in zip(bars, rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontweight='bold')
ax.set_title('STD Positive Rate by Condom Use')
ax.set_ylabel('STD Positive Rate (%)')
plt.tight_layout()
plt.savefig('plots/03_std_by_condom_use.png', dpi=150)
plt.close()

# ── Plot 4: STD Rate by Number of Partners ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
rate = df.groupby('num_partners')['std_positive'].mean() * 100
ax.bar(rate.index, rate.values, color='purple', alpha=0.75)
ax.plot(rate.index, rate.values, 'k-o', markersize=5)
ax.set_title('STD Positive Rate by Number of Partners')
ax.set_xlabel('Number of Sexual Partners')
ax.set_ylabel('STD Positive Rate (%)')
ax.set_xticks(rate.index)
plt.tight_layout()
plt.savefig('plots/04_std_by_partners.png', dpi=150)
plt.close()

# ── Plot 5: STD Rate by Gender ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cat_cols = ['gender', 'alcohol_use', 'relationship_status']
titles   = ['By Gender', 'By Alcohol Use', 'By Relationship Status']
colors   = [['steelblue','crimson'], ['green','orange','red'], ['teal','purple','brown','gray']]
for ax, col, title, clrs in zip(axes, cat_cols, titles, colors):
    rate = df.groupby(col)['std_positive'].mean() * 100
    ax.bar(rate.index, rate.values, color=clrs[:len(rate)], alpha=0.85)
    for i, (idx, val) in enumerate(rate.items()):
        ax.text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=9)
    ax.set_title(f'STD Rate {title}')
    ax.set_ylabel('STD Positive Rate (%)')
plt.suptitle('STD Rate by Demographic Factors', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/05_std_by_demographics.png', dpi=150)
plt.close()

# ── Plot 6: Symptom Prevalence ────────────────────────────────────────────────
symptoms = ['symptom_discharge','symptom_sores','symptom_burning','symptom_rash']
labels   = ['Discharge','Sores','Burning','Rash']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, label, color in zip(axes, ['STD Negative (0)','STD Positive (1)'], ['steelblue','crimson']):
    group = df[df['std_positive'] == (1 if 'Positive' in label else 0)]
    rates = [group[s].mean() * 100 for s in symptoms]
    ax.barh(labels, rates, color=color, alpha=0.8)
    ax.set_xlabel('% with Symptom')
    ax.set_title(label)
    ax.set_xlim(0, 60)
    for i, v in enumerate(rates):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
plt.suptitle('Symptom Prevalence: Positive vs Negative', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/06_symptom_prevalence.png', dpi=150)
plt.close()

# ── Plot 7: STD Rate by Education & Testing Frequency ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
edu_order  = ['none','primary','secondary','tertiary']
test_order = ['never','yearly','every_6_months','every_3_months']
for ax, col, order, title in zip(
    axes,
    ['education', 'sti_test_freq'],
    [edu_order, test_order],
    ['Education Level', 'STI Testing Frequency']
):
    rate = df.groupby(col)['std_positive'].mean().reindex(order) * 100
    bars = ax.bar(order, rate.values, color='teal', alpha=0.8)
    for bar, val in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', fontsize=9)
    ax.set_title(f'STD Rate by {title}')
    ax.set_ylabel('STD Positive Rate (%)')
    ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('plots/07_education_testing.png', dpi=150)
plt.close()

# ── Plot 8: Correlation Heatmap ────────────────────────────────────────────────
num_cols = ['age','num_partners','prev_std','iv_drug_use','hiv_positive',
            'symptom_discharge','symptom_sores','symptom_burning','symptom_rash',
            'age_first_sex','std_positive']
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/08_correlation_heatmap.png', dpi=150)
plt.close()

print("\nAll EDA plots saved to plots/")
