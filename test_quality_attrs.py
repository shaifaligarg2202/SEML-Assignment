"""
Quality-attribute verification suite for the Loan Approval Risk Service.

Each test maps to one of the top quality requirements and to a property of the
Pipe-and-Filter architecture:

  Robustness      -> schema bounds + business-rule rejection
  Reliability     -> output conforms to the JSON response schema
  Performance     -> average pipeline latency under the SLA
  Maintainability -> filters are pure / independently testable

Run:  python -m pytest tests/ -v
      python -m unittest tests.test_quality_attrs -v
"""
import os
import time
import unittest

from pydantic import ValidationError

from app.schemas import LoanApplicationInput, LoanApprovalResult
from app.pipeline import (
    validate_input,
    extract_features,
    run_model,
    format_response,
    execute_pipeline,
)

LATENCY_SLA_MS = 150.0


# A valid, low-risk applicant used as the baseline across tests.
VALID_PAYLOAD = dict(
    age=45, annual_income=95000, credit_score=720, employment_status=0,
    education_level=4, loan_amount=25000, loan_duration=36,
    monthly_debt_payments=400, credit_card_utilization_rate=0.25,
    debt_to_income_ratio=0.15, bankruptcy_history=0, loan_purpose=3,
    previous_loan_defaults=0, payment_history=28, length_of_credit_history=15,
    savings_account_balance=35000, checking_account_balance=8000,
    total_liabilities=30000, job_tenure=10, net_worth=220000,
)


class _StubModel:
    """Deterministic stand-in classifier so filter tests never require a
    trained artifact. Returns P(approve) as a smooth function of credit score."""

    def predict_proba(self, X):
        # X is a DataFrame; column 2 is CreditScore per settings.FEATURES order.
        score = float(X.iloc[0, 2])
        p_approve = max(0.01, min(0.99, (score - 300) / 550.0))
        return [[1.0 - p_approve, p_approve]]


def _load_model():
    """Use the real serialized model when available; otherwise the stub."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "model.pkl"
    )
    if os.path.exists(model_path):
        try:
            import joblib
            return joblib.load(model_path)
        except Exception:
            pass
    return _StubModel()


class TestQualityAttributes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _load_model()

    # ── Robustness: schema boundary validation ───────────────────────
    def test_schema_rejects_low_age(self):
        """Applicants under 18 must be rejected at the schema boundary."""
        bad = {**VALID_PAYLOAD, "age": 16}
        with self.assertRaises(ValidationError):
            LoanApplicationInput(**bad)

    def test_schema_rejects_invalid_credit_score(self):
        """Credit score outside [300, 850] must be rejected at the boundary."""
        with self.assertRaises(ValidationError):
            LoanApplicationInput(**{**VALID_PAYLOAD, "credit_score": 950})
        with self.assertRaises(ValidationError):
            LoanApplicationInput(**{**VALID_PAYLOAD, "credit_score": 250})

    # ── Robustness: business-rule rejection in Filter 1 ──────────────
    def test_pipeline_rejects_excessive_loan_ratio(self):
        """Loan amount > 500% of annual income is a business-rule failure."""
        risky = LoanApplicationInput(
            **{**VALID_PAYLOAD, "annual_income": 20000, "loan_amount": 150000}
        )
        with self.assertRaises(ValueError):
            validate_input(risky)

    # ── Reliability: output conforms to the response schema ──────────
    def test_pipeline_output_conforms_to_schema(self):
        """The full pipeline must return a well-formed LoanApprovalResult."""
        app_in = LoanApplicationInput(**VALID_PAYLOAD)
        result = execute_pipeline(app_in, self.model)
        self.assertIsInstance(result, LoanApprovalResult)
        self.assertIsInstance(result.is_approved, bool)
        self.assertIn(result.risk_tier, {"LOW", "MEDIUM", "HIGH"})
        self.assertGreaterEqual(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)

    # ── Performance: latency under SLA ───────────────────────────────
    def test_inference_pipeline_latency(self):
        """Average pipeline latency must stay under the 150 ms SLA."""
        app_in = LoanApplicationInput(**VALID_PAYLOAD)
        runs = 50
        start = time.perf_counter()
        for _ in range(runs):
            execute_pipeline(app_in, self.model)
        avg_ms = (time.perf_counter() - start) / runs * 1000.0
        self.assertLess(
            avg_ms, LATENCY_SLA_MS,
            f"Average latency {avg_ms:.2f} ms exceeded SLA {LATENCY_SLA_MS} ms",
        )

    # ── Maintainability: Filter 1 is a pure function ─────────────────
    def test_validation_filter_is_pure(self):
        """validate_input must not mutate its input and must be idempotent."""
        app_in = LoanApplicationInput(**VALID_PAYLOAD)
        snapshot = app_in.model_dump()
        out1 = validate_input(app_in)
        out2 = validate_input(app_in)
        self.assertEqual(app_in.model_dump(), snapshot)      # no mutation
        self.assertEqual(out1.model_dump(), out2.model_dump())  # idempotent

    # ── Maintainability: Filter 2 is isolated / deterministic ────────
    def test_feature_extraction_filter_isolation(self):
        """extract_features must be deterministic and shape-correct in isolation."""
        from app.config import settings
        app_in = LoanApplicationInput(**VALID_PAYLOAD)
        fv1, nw1, dti1 = extract_features(app_in)
        fv2, nw2, dti2 = extract_features(app_in)
        self.assertEqual(fv1.shape, (1, len(settings.FEATURES)))
        self.assertTrue((fv1.values == fv2.values).all())    # deterministic
        self.assertEqual(nw1, app_in.net_worth)


if __name__ == "__main__":
    unittest.main(verbosity=2)
