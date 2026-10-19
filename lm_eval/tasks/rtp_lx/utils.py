"""Data loading for RTP-LX.

RTP-LX ships only as a password-protected zip on GitHub, so it cannot be pulled
from the HF Hub inside a job. The data is prepared once with
``multilingual-harmfulness/benchmarks/rtp_lx/prepare_data.py`` into
``<RTP_LX_DATA_DIR>/<LOCALE>.jsonl`` (one file per locale), and loaded here.
"""

import os

import datasets


DEFAULT_DATA_DIR = "/capstor/scratch/cscs/alexandersternfeld/mh-data/rtp_lx/processed"
DATA_DIR_ENV = "RTP_LX_DATA_DIR"


def data_dir() -> str:
    return os.getenv(DATA_DIR_ENV) or DEFAULT_DATA_DIR


def load_rtp_lx(locale: str, **kwargs):
    """custom_dataset hook: returns {"test": Dataset} for one locale."""
    path = os.path.join(data_dir(), f"{locale}.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"RTP-LX data for locale {locale!r} not found at {path}. Run "
            "benchmarks/rtp_lx/prepare_data.py or set "
            f"{DATA_DIR_ENV} to the directory holding <LOCALE>.jsonl files."
        )
    ds = datasets.load_dataset("json", data_files={"test": path}, split="test")
    return {"test": ds}


def task_suffix(locale: str) -> str:
    return locale.lower().replace("-", "_")
