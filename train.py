"""
STD Prediction — Model Training & Evaluation
Trains multiple ML models, evaluates, and saves the best one.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ── Load & encode ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/std_dataset.csv')

encoders = {}
cat_cols = ['gender', 'condom_use', 'alcohol_use', 'education',
            'sti_test_freq', 'urban_rural', 'relationship_status']

for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    encoders[col] = le

features = [
    'age', 'gender_enc', 'num_partners', 'condom_use_enc', 'prev_std',
    'iv_drug_use', 'alcohol_use_enc', 'education_enc', 'hiv_positive',
    'sti_test_freq_enc', 'urban_rural_enc', 'relationship_status_enc',
    'symptom_discharge', 'symptom_sores', 'symptom_burning', 'symptom_rash',
    'age_first_sex'
]

X = df[features]
y = df['std_positive']

# ── Split ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── SMOTE to balance classes ───────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Negative: {(y_train_bal==0).sum():,} | Positive: {(y_train_bal==1).sum():,}")

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc  = scaler.transform(X_test)

# ── Models ─────────────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n── Model Comparison ─────────────────────────────────────────────────────")
print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'AUC':>7}")
print("-"*70)

for name, model in models.items():
    use_scaled = name == 'Logistic Regression'
    Xtr = X_train_sc if use_scaled else X_train_bal
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train_bal)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model': model, 'scaled': use_scaled,
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc,
        'y_pred': y_pred, 'y_prob': y_prob
    }
    print(f"{name:<25} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>7.3f} {auc:>7.3f}")

# ── Best model by F1 ───────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
best = results[best_name]
print(f"\nBest model: {best_name} (F1: {best['f1']:.3f} | AUC: {best['auc']:.3f})")
print("\nDetailed Classification Report:")
print(classification_report(y_test, best['y_pred'], target_names=['Negative', 'Positive']))

# ── Plot 9: Confusion Matrix ───────────────────────────────────────────────────
cm = confusion_matrix(y_test, best['y_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'], ax=ax)
ax.set_title(f'Confusion Matrix — {best_name}')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/09_confusion_matrix.png', dpi=150)
plt.close()

# ── Plot 10: ROC Curves ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — STD Prediction Models')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('plots/10_roc_curves.png', dpi=150)
plt.close()

# ── Plot 11: Precision-Recall ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results.items():
    p, r, _ = precision_recall_curve(y_test, res['y_prob'])
    ap = average_precision_score(y_test, res['y_prob'])
    ax.plot(r, p, label=f"{name} (AP={ap:.3f})")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — STD Prediction')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('plots/11_precision_recall.png', dpi=150)
plt.close()

# ── Plot 12: Feature Importance ────────────────────────────────────────────────
if hasattr(best['model'], 'feature_importances_'):
    imp = pd.Series(best['model'].feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.plot(kind='barh', color='steelblue', ax=ax)
    ax.set_title(f'Feature Importance — {best_name}')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('plots/12_feature_importance.png', dpi=150)
    plt.close()

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump(best['model'], 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoders.pkl')
joblib.dump({'best_model': best_name, 'features': features, 'scaled': best['scaled']},
            'models/metadata.pkl')

print(f"\nModel saved → models/best_model.pkl")
print(f"Plots saved → plots/")
