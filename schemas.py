from pydantic import BaseModel, Field


class LoanApplicationInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the applicant")
    annual_income: int = Field(..., gt=0, description="Applicant's yearly income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_status: int = Field(..., ge=0, le=2, description="0=Employed, 1=Self-Employed, 2=Unemployed")
    education_level: int = Field(..., ge=0, le=4, description="0=Associate, 1=Bachelor, 2=Doctorate, 3=High School, 4=Master")
    loan_amount: int = Field(..., gt=0, description="Requested loan amount")
    loan_duration: int = Field(..., ge=12, le=120, description="Loan term in months")
    monthly_debt_payments: int = Field(..., ge=0, description="Monthly debt payments")
    credit_card_utilization_rate: float = Field(..., ge=0.0, le=1.0, description="Credit card utilization (0.0-1.0)")
    debt_to_income_ratio: float = Field(..., ge=0.0, description="Debt-to-income ratio")
    bankruptcy_history: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    loan_purpose: int = Field(..., ge=0, le=4, description="0=Auto, 1=Debt Consolidation, 2=Education, 3=Home, 4=Other")
    previous_loan_defaults: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    payment_history: int = Field(..., ge=0, description="Payment history score")
    length_of_credit_history: int = Field(..., ge=0, description="Credit history in years")
    savings_account_balance: int = Field(..., ge=0, description="Savings account balance")
    checking_account_balance: int = Field(..., ge=0, description="Checking account balance")
    total_liabilities: int = Field(..., ge=0, description="Total liabilities")
    job_tenure: int = Field(..., ge=0, description="Job tenure in years")
    net_worth: int = Field(..., description="Net worth (assets - liabilities)")


class LoanApprovalResult(BaseModel):
    is_approved: bool = Field(..., description="Binary classification outcome")
    probability: float = Field(..., description="Model probability of loan approval")
    risk_tier: str = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    net_worth: int = Field(..., description="Applicant's net worth")
    debt_to_income: float = Field(..., description="Debt-to-income ratio")
    latency_ms: float = Field(..., description="API processing time in milliseconds")
