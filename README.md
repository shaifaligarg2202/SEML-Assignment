# Loan Underwriting & Credit Risk Service (Group 84)

This repository contains the implementation of the **Loan Approval Risk Assessment Service**, designed for automated consumer lending credit-risk evaluation.

The project applies **Software Engineering for ML (SE4ML)** best practices, integrating **GR4ML requirements modeling** with two architectural patterns — a **Pipe-and-Filter** inference pipeline served as a **FastAPI microservice** with structured event logging.

---

## Project Structure

```
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml                 # Externalized system + model thresholds
├── data/
│   ├── generate_synthetic_data.py  # Schema-faithful stand-in for the Kaggle file
│   ├── prepare_data.py             # Feature engineering & preprocessing
│   ├── Loan.csv                    # Raw dataset, 20,000 rows × 36 cols (gitignored)
│   └── loan_data_processed.csv     # Processed dataset, 22 features + target (gitignored)
├── app/
│   ├── __init__.py
│   ├── config.py                   # PyYAML-based configuration management
│   ├── schemas.py                  # Pydantic input/output schemas (robustness boundary)
│   ├── pipeline.py                 # Pipe-and-Filter execution engine (4 filters)
│   ├── logger.py                   # Structured JSON logger & runtime timer
│   ├── main.py                     # FastAPI serving microservice
│   └── model.pkl                   # Serialized trained RandomForest (gitignored)
├── training/
│   ├── __init__.py
│   └── step1_notebook.py           # RandomForest vs LogisticRegression training
├── tests/
│   ├── __init__.py
│   └── test_quality_attrs.py       # 7 quality-attribute unit tests
├── diagrams/
│   ├── gr4ml_business_view.png
│   ├── gr4ml_analytics_design_view.png
│   ├── gr4ml_data_prep_view.png
│   ├── system_architecture.png
│   ├── swagger_request.png
│   └── swagger_response.png
├── generate_gr4ml_diagrams.py      # Regenerates all four diagrams
└── Group_84.ipynb                  # Jupyter Notebook report
```

---

## Architecture Design & Patterns

### 1. Pipe-and-Filter Pattern (`app/pipeline.py`)
The runtime prediction sequence is a series of decoupled Filters connected by Pipes:
* **Filter 1 (`validate_input`):** Business-rule validation (e.g. loan ≤ 500% of annual income). *Robustness.*
* **Filter 2 (`extract_features`):** Computes derived features (`LoanToIncomeRatio`, `SavingsToLoanRatio`) and builds the model DataFrame. *Maintainability.*
* **Filter 3 (`run_model`):** Invokes model inference and applies the decision threshold. *Performance.*
* **Filter 4 (`format_response`):** Maps probability → risk tier (LOW/MEDIUM/HIGH) and packages the output. *Reliability.*

### 2. Microservice Serving & Event Logging (`app/main.py`)
* The pipeline is wrapped in a **FastAPI** service exposing `POST /predict` and `GET /health`.
* Every transaction is logged in **structured JSON** (latency, credit score, approval, risk tier) for monitoring.

---

## Dataset

**Source:** Kaggle — *Financial Risk for Loan Approval* (`lorenzozoppelletto/financial-risk-for-loan-approval`)
- 20,000 records, 36 original columns.
- After feature engineering: **22 features + 1 target** (`LoanApproved`).

Because the Kaggle file is licensed and not redistributed here, `data/generate_synthetic_data.py`
produces a **schema-identical stand-in with a genuine, learnable signal**, so the entire pipeline is
reproducible without the private download. Drop in the real `data/Loan.csv` to use the original data.

### Feature Engineering Pipeline (`data/prepare_data.py`)
1. Drop 6 non-predictive columns (`ApplicationDate`, `RiskScore`, `InterestRate`, `BaseInterestRate`, `MonthlyLoanPayment`, `TotalDebtToIncomeRatio`) → 36 → 30
2. Drop 3 redundant features via correlation r > 0.95 (`MonthlyIncome`, `Experience`, `TotalAssets`) → 30 → 27
3. Drop 6 low-importance features (`MaritalStatus`, `HomeOwnershipStatus`, …) → 27 → 21
4. Encode categoricals (`EmploymentStatus`, `EducationLevel`, `LoanPurpose`)
5. Engineer 2 derived ratios (`LoanToIncomeRatio`, `SavingsToLoanRatio`) → 21 → 23 columns (**22 features + target**)

---

## Quick Start

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Data  (generate synthetic stand-in, or supply the real data/Loan.csv)
python data/generate_synthetic_data.py
python data/prepare_data.py

# 3. Train  (RandomForest + LogisticRegression comparison, saves app/model.pkl)
python training/step1_notebook.py

# 4. Verify quality attributes  (7 tests)
python -m pytest tests/ -v

# 5. Serve
uvicorn app.main:app --reload --port 8000
#    -> open http://127.0.0.1:8000/docs  for Swagger UI

# 6. (optional) Regenerate all diagrams
python generate_gr4ml_diagrams.py
```

---

## Quality Attributes (verified by `tests/test_quality_attrs.py`)

| Attribute | Implementation | Verifying test |
|-----------|----------------|----------------|
| **Robustness** | Pydantic schema bounds + business-rule checks in Filter 1 | `test_schema_rejects_*`, `test_pipeline_rejects_excessive_loan_ratio` |
| **Reliability** | Deterministic risk-tier output conforming to a JSON schema | `test_pipeline_output_conforms_to_schema` |
| **Performance** | Latency tracked against a <150 ms SLA | `test_inference_pipeline_latency` |
| **Maintainability** | Pure, side-effect-free filters testable in isolation | `test_validation_filter_is_pure`, `test_feature_extraction_filter_isolation` |
