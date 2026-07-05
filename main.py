import os
import joblib
from fastapi import FastAPI, HTTPException
from app.schemas import LoanApplicationInput, LoanApprovalResult
from app.config import settings
from app.pipeline import execute_pipeline
from app.logger import logger, Timer

app = FastAPI(title=settings.PROJECT_NAME)

# Global model store representation
class ModelStore:
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(settings.MODEL_PATH):
            try:
                self.model = joblib.load(settings.MODEL_PATH)
                print(f"Model loaded successfully from {settings.MODEL_PATH}")
            except Exception as e:
                print(f"Error loading model from {settings.MODEL_PATH}: {str(e)}")
        else:
            print(f"[WARNING] Model file not found at {settings.MODEL_PATH}. Prediction requests will fail.")

store = ModelStore()

@app.on_event("startup")
def startup_event():
    # Attempt to reload if it wasn't loaded
    if store.model is None:
        store.load_model()

@app.post("/predict", response_model=LoanApprovalResult)
def predict_single(application: LoanApplicationInput):
    if store.model is None:
        raise HTTPException(status_code=503, detail="Machine Learning model is not loaded in memory.")
        
    try:
        with Timer() as t:
            result = execute_pipeline(application, store.model)
            
        # Log the transaction with structured extra fields
        logger.info(
            "loan_evaluation_served", 
            extra={
                "latency_ms": round(t.elapsed_ms, 2),
                "credit_score": application.credit_score,
                "loan_amount": application.loan_amount,
                "is_approved": result.is_approved,
                "probability": result.probability,
                "risk_tier": result.risk_tier
            }
        )
        return result
        
    except ValueError as val_err:
        # Client validation error / business rule failure
        logger.warning(f"Business rule validation failed: {str(val_err)}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logger.error(f"Inference pipeline execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal pipeline error: {str(e)}")

@app.get("/health")
def health_check():
    model_status = "loaded" if store.model is not None else "missing"
    return {
        "status": "healthy" if store.model is not None else "degraded",
        "model_status": model_status,
        "environment": settings.APP_ENV,
        "features_configured": len(settings.FEATURES)
    }
