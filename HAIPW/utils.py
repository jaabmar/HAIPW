import logging
import pandas as pd
import numpy as np
import os


def setup_logging(log_file: str):
    """Configures logging to output to both console and a file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger("httpx").setLevel(logging.WARNING)


def log(message):
    logging.info(message)


def load_dataframe(file_path):
    return pd.read_csv(file_path)


def compute_coverage(est, var, gt, n_rct, zalpha=1.96):
    margin = zalpha * np.sqrt(var / n_rct)
    return int((est - margin < gt) and (gt < est + margin)) * 100
