import os
import yaml

# Dynamically resolve the project root directory (parent of 'app/')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings:
    def __init__(self):
        self.PROJECT_NAME = "Loan Approval Risk Service"
        self.APP_ENV = "production"
        self.MODEL_PATH = os.path.join(BASE_DIR, "app", "model.pkl")
        self.DEFAULT_DECISION_THRESHOLD = 0.50
        self.LOG_LEVEL = "INFO"
        self.FEATURES = []

        # Load from config.yaml if available
        config_path = os.path.join(BASE_DIR, "configs", "config.yaml")

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.PROJECT_NAME = config['app']['name']
                    self.APP_ENV = config['app']['env']
                    self.MODEL_PATH = os.path.join(BASE_DIR, config['model']['path'])
                    self.DEFAULT_DECISION_THRESHOLD = float(config['model']['default_threshold'])
                    self.FEATURES = config['model']['features']
                    self.LOG_LEVEL = config['logging']['level']
                    print(f"Configurations loaded successfully from {config_path}")
            except Exception as e:
                print(f"[WARNING] Failed to parse config.yaml, using defaults. Error: {str(e)}")
        else:
            print(f"[WARNING] config.yaml not found at {config_path}, using code defaults.")
            self.FEATURES = [
                'Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus',
                'EducationLevel', 'LoanAmount', 'LoanDuration',
                'MonthlyDebtPayments', 'CreditCardUtilizationRate',
                'DebtToIncomeRatio', 'BankruptcyHistory', 'LoanPurpose',
                'PreviousLoanDefaults', 'PaymentHistory', 'LengthOfCreditHistory',
                'SavingsAccountBalance', 'CheckingAccountBalance',
                'TotalLiabilities', 'JobTenure', 'NetWorth',
                'LoanToIncomeRatio', 'SavingsToLoanRatio'
            ]

settings = Settings()
