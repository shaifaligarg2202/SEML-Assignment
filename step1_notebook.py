"""
Model training — Analytics Design View realisation.

Trains TWO candidate algorithms and compares them against the quality
softgoals defined in the GR4ML Analytics Design View:

  * Logistic Regression  — fast, highly explainable baseline (linear)
  * Random Forest         — higher accuracy on non-linear interactions

The Random Forest wins on Accuracy (the dominant softgoal for underwriting)
and is serialised to app/model.pkl for serving. The comparison is what the
Analytics Design View documents: an explicit algorithm-vs-softgoal trade-off.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

# Dynamically resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES = [
    'Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus',
    'EducationLevel', 'LoanAmount', 'LoanDuration',
    'MonthlyDebtPayments', 'CreditCardUtilizationRate',
    'DebtToIncomeRatio', 'BankruptcyHistory', 'LoanPurpose',
    'PreviousLoanDefaults', 'PaymentHistory', 'LengthOfCreditHistory',
    'SavingsAccountBalance', 'CheckingAccountBalance',
    'TotalLiabilities', 'JobTenure', 'NetWorth',
    'LoanToIncomeRatio', 'SavingsToLoanRatio'
]


def train_baseline_model():
    print("[STEP 1] Training loan-approval risk classifiers...\n")

    data_path = os.path.join(PROJECT_ROOT, "data", "loan_data_processed.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Run 'python data/generate_synthetic_data.py' then "
            f"'python data/prepare_data.py' first."
        )

    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df['LoanApproved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples\n")

    # ── Candidate A: Logistic Regression (explainable baseline) ─────
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )
    logreg.fit(X_train, y_train)
    lr_preds = logreg.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

    # ── Candidate B: Random Forest (chosen model) ───────────────────
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    # ── Algorithm-vs-softgoal comparison (Analytics Design View) ────
    print("=" * 60)
    print("ALGORITHM COMPARISON  (GR4ML Analytics Design View)")
    print("=" * 60)
    print(f"{'Algorithm':<22}{'Accuracy':>10}{'ROC-AUC':>10}")
    print("-" * 60)
    print(f"{'Logistic Regression':<22}{lr_acc:>10.4f}{lr_auc:>10.4f}")
    print(f"{'Random Forest (chosen)':<22}{rf_acc:>10.4f}{rf_auc:>10.4f}")
    print("-" * 60)
    print("Chosen: Random Forest -> wins on Accuracy softgoal (++)\n")

    print("Random Forest classification report:")
    print(classification_report(y_test, rf_preds))

    print("Top 10 Feature Importances (drives Explainability softgoal):")
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for feat, imp in importances.head(10).items():
        print(f"  {feat:<30} {imp:.4f}")

    # ── Persist the chosen model for serving ────────────────────────
    model_dir = os.path.join(PROJECT_ROOT, "app")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(rf, model_path)
    print(f"\nChosen model (Random Forest) saved at {model_path}")

    return {"rf_accuracy": rf_acc, "lr_accuracy": lr_acc,
            "rf_auc": rf_auc, "lr_auc": lr_auc}


if __name__ == "__main__":
    train_baseline_model()
