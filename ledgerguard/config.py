"""Tunable thresholds for every detection rule."""
from dataclasses import dataclass


@dataclass
class Config:
    # Control environment
    approval_limit: float = 10_000.0      # entries >= this need a second approver
    threshold_band: float = 0.05          # flag amounts within 5% below the limit
    # Round-number rule
    round_min: float = 1_000.0            # only flag round amounts >= this
    # Posting-time rule
    business_start: int = 7               # first normal posting hour (inclusive)
    business_end: int = 19                # first abnormal posting hour (exclusive)
    # Duplicate rule
    duplicate_window_days: int = 7
    # Outlier rule (robust z-score on log amounts, per account)
    outlier_z: float = 4.0
    outlier_min_n: int = 20
    # Vendor-level Benford rule
    benford_min_n: int = 30
    benford_chi2: float = 26.125          # chi-square critical value, df=8, p=0.001
    benford_excess: float = 0.05          # digit must be 5pp above expected to flag entries
    # Risk tiers
    high_threshold: float = 0.65
    medium_threshold: float = 0.40
