import logging
import json
import sys
import time

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        # Add extra fields if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_record.update(record.extra)
        elif hasattr(record, "__dict__"):
            # Fetch custom extra fields added in logging.info
            for key, val in record.__dict__.items():
                if key not in [
                    "args", "asctime", "created", "exc_info", "exc_text", 
                    "filename", "funcName", "levelname", "levelno", "lineno", 
                    "module", "msecs", "msg", "name", "pathname", "process",
                    "processName", "relativeCreated", "stack_info", "thread",
                    "threadName", "taskName"
                ]:
                    log_record[key] = val
                    
        return json.dumps(log_record)

def setup_logger():
    logger = logging.getLogger("loan_service")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed_ms = (self.end - self.start) * 1000.0
