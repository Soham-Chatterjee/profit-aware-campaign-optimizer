"""Central configuration: paths, data contract, feature policy and eligibility rules.

Every assumption lives here so that notebooks, the pipeline and the Streamlit app
share one source of truth. Later milestones add economics, channel and GA settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_CSV = RAW_DIR / "bank-full.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_CSV_SHA256 = "D1513EC63B385506F7CFCE9F2C5CAA9FE99E7BA4E8C3FA264B3AAF0F849ED32D"

# --------------------------------------------------------------------------- data contract
EXPECTED_ROWS = 45_211
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y",
]
CATEGORICAL_RAW = ["job", "marital", "education", "contact", "month", "poutcome"]
BINARY_RAW = ["default", "housing", "loan", "y"]
TARGET = "y"

# --------------------------------------------------------------------------- leakage policy (D-004)
# Columns that are only known once the campaign has started. Never allowed as model features.
LEAKAGE_COLUMNS = frozenset({"duration", "campaign", "day", "month"})
FORBIDDEN_FEATURES = LEAKAGE_COLUMNS | {TARGET}

# --------------------------------------------------------------------------- split / universe (D-014)
RANDOM_STATE = 42
TEST_SIZE = 0.30
UNIVERSE_SIZE = 10_000
CUSTOMER_ID_START = 100_001  # IDs are C100001, C100002, ...


# --------------------------------------------------------------------------- eligibility (D-007)
@dataclass(frozen=True)
class EligibilityConfig:
    exclude_default: bool = True
    exclude_negative_balance: bool = True
    exclude_fatigue: bool = True
    fatigue_max_previous: int = 10      # previous > this => excluded
    protected_age: int = 75             # age >= this => flagged (capped in optimisation, not excluded)


ELIGIBILITY = EligibilityConfig()


# --------------------------------------------------------------------------- feature engineering
@dataclass(frozen=True)
class FeatureConfig:
    age_bins: tuple = (17, 29, 44, 59, 74, 200)
    age_labels: tuple = ("18-29", "30-44", "45-59", "60-74", "75+")
    recency_bins: tuple = (0, 90, 180, 365, 10_000)
    recency_labels: tuple = ("<=90d", "91-180d", "181-365d", ">365d")
    frequency_cap: int = 10            # previous is clipped at this value for modelling
    balance_quintiles: int = 5


FEATURES = FeatureConfig()


# --------------------------------------------------------------------------- channels & budget (D-009, D-010)
# Channel is fixed per customer from the `contact` field; cost is fully loaded per contact.
CHANNEL_MAP = {"cellular": "mobile_call", "telephone": "landline_call", "unknown": "digital"}
CHANNEL_COST = {"mobile_call": 12.0, "landline_call": 10.0, "digital": 2.0}
CAMPAIGN_BUDGET = 50_000.0
# Campaign capacity (used by the M3 budget-binding check and by the M4 optimiser).
CHANNEL_GROUP = {"mobile_call": "calls", "landline_call": "calls", "digital": "digital"}
CHANNEL_CAPACITY = {"calls": 4_500, "digital": 2_500}
MAX_CONTACTS = 6_000


# --------------------------------------------------------------------------- product economics (D-008, D-031)
@dataclass(frozen=True)
class EconomicsConfig:
    """Term deposit valued as a funding product. The principal is never revenue.

    D_i  = clip(deposit_min + deposit_alpha * max(balance_i, 0), deposit_min, deposit_max)
    V_i  = D_i * (ftp_rate - customer_rate) * term_years - incentive - onboarding_cost
    EP_i = (P_i - organic_rate) * V_i - C_i;   BE_i = (P_i - organic_rate) * V_i
    Defaults are illustrative and calibrated in M3 (D-031), not bank figures.
    """
    deposit_min: float = 10_000.0
    deposit_alpha: float = 1.0
    deposit_max: float = 100_000.0
    ftp_rate: float = 0.0475
    customer_rate: float = 0.0300
    term_years: float = 1.0
    incentive: float = 40.0
    onboarding_cost: float = 5.0
    organic_rate: float = 0.0
    channel_cost: dict = field(default_factory=lambda: dict(CHANNEL_COST))

    @property
    def spread(self) -> float:
        return self.ftp_rate - self.customer_rate


ECONOMICS = EconomicsConfig()


# --------------------------------------------------------------------------- segmentation (D-013, D-032, D-033)
SEGMENTS = ("High value", "Growing", "Dormant", "Low engagement")


@dataclass(frozen=True)
class SegmentConfig:
    high_propensity_quantile: float = 0.60    # top 40% of propensity = "high"
    high_value_quantile: float = 0.50         # V_i at or above the median = "high"
    # RFM scores 1-5 on fixed, business-readable bins (D-033)
    recency_score: tuple = (("never", 1), (">365d", 2), ("181-365d", 3), ("91-180d", 4), ("<=90d", 5))
    frequency_bins: tuple = (-1, 0, 1, 2, 4, 10_000)        # previous: 0 | 1 | 2 | 3-4 | 5+
    engagement_weights: tuple = (0.4, 0.3, 0.3)             # recency, frequency, prior success


SEGMENTATION = SegmentConfig()


# --------------------------------------------------------------------------- campaign constraints (plan M4)
@dataclass(frozen=True)
class CampaignConstraints:
    """Limits every strategy must respect. Shares are of the selected list (count-based)."""
    budget: float = CAMPAIGN_BUDGET
    max_contacts: int = MAX_CONTACTS
    channel_capacity: dict = field(default_factory=lambda: dict(CHANNEL_CAPACITY))
    segment_max_share: dict = field(default_factory=lambda: {s: 0.60 for s in SEGMENTS})
    segment_min_share: dict = field(default_factory=lambda: {"Growing": 0.15, "Dormant": 0.10})
    protected_max_share: float = 0.02        # customers aged 75+ (flagged, D-007)


CONSTRAINTS = CampaignConstraints()


# --------------------------------------------------------------------------- genetic algorithm (D-035 .. D-037)
@dataclass(frozen=True)
class GAConfig:
    population: int = 100
    generations: int = 300
    patience: int = 50                 # stop after this many generations without improvement
    tournament_k: int = 3
    crossover_rate: float = 0.9        # uniform crossover
    # Asymmetric bit-flip (D-036): many adds, few drops. Repair then drops the weakest selections,
    # so a mutation acts as a set of swaps instead of a random walk away from feasibility.
    mutation_adds: float = 30.0        # expected genes switched ON per child
    mutation_drops: float = 0.5        # expected genes switched OFF per child
    elite_frac: float = 0.02
    greedy_seeds: int = 4              # 0 = unseeded run
    fill: bool = True                  # repair also tops up unused capacity (D-036)
    time_limit_s: float | None = None  # optional wall-clock cap (e.g. for the dashboard)
    random_state: int = RANDOM_STATE


GA = GAConfig()


@dataclass(frozen=True)
class OptimConfig:
    random_runs: int = 50              # Random baseline averaged over this many seeds
    milp_time_limit: float = 180.0     # seconds
    milp_rel_gap: float = 1e-6


OPTIM = OptimConfig()


# --------------------------------------------------------------------------- propensity model (D-006, D-026)
@dataclass(frozen=True)
class ModelConfig:
    selected: str = "lr"                                   # "lr" (primary) or "rf" (challenger)
    lr_c_grid: tuple = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    lr_calibration: str | None = "isotonic"               # D-027: fixes top-decile under-prediction
    propensity_cap: float = 0.90                           # D-028: guard against isotonic tail over-fit
    parallel_backend: str = "threading"                    # D-029: loky workers break without a console on Windows
    n_jobs: int = -1
    cv_folds: int = 5
    rf_n_estimators: int = 300
    rf_min_samples_leaf: int = 50
    rf_max_features: str = "sqrt"
    rf_calibration: str = "isotonic"
    precision_k: tuple = (500, 1_000, 2_000)               # plus the budget-implied K
    switch_threshold_pr_auc: float = 0.02                  # RF must beat LR by this to be preferred


MODEL = ModelConfig()


@dataclass
class Settings:
    """Bundle of all configuration objects (extended in later milestones)."""
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    segmentation: SegmentConfig = field(default_factory=SegmentConfig)
    constraints: CampaignConstraints = field(default_factory=CampaignConstraints)
    ga: GAConfig = field(default_factory=GAConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
