import os
import logging
import sys

def get_logger(name: str = "authenticity", log_dir: str = None) -> logging.Logger:
    if log_dir is None:
        log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "logs", "authenticity")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)
        
        log_file = os.path.join(log_dir, "authenticity.log")
        f_handler = logging.FileHandler(log_file, encoding="utf-8")
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s [%(filename)s:%(lineno)d]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
        
    return logger
