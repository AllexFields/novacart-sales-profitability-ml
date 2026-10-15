"""
NovaCart AI Business Intelligence System - Feature Engineering Pipeline
Constructs domain-specific features, bucketed representations, and interaction terms
both as standalone dataset transformers and as an enterprise Scikit-Learn Pipeline Transformer.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Support direct execution and package-level imports
try:
    from src.utils import DATA_DIR, setup_logger
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.utils import DATA_DIR, setup_logger

logger = setup_logger("feature_engineering")

# Core inference input features provided before order completion
RAW_INPUT_FEATURES = [
    "Region",
    "Category",
    "Sales",
    "Discount_pct",
    "Delivery_Days",
    "Customer_Rating",
]

# Engineered feature names produced for downstream model ingestion
ENGINEERED_NUMERIC_FEATURES = [
    "Sales",
    "Discount_pct",
    "Delivery_Days",
    "Customer_Rating",
    "High_Discount_Flag",
    "Late_Delivery_Flag",
    "Low_Rating_Flag",
    "High_Risk_Combination_Flag",
    "Effective_Sales",
    "Discount_Amount",
    "Rating_Delivery_Ratio",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "Region",
    "Category",
    "Discount_Bucket",
    "Sales_Bucket",
    "Region_Category",
    "Delivery_Category",
]

ALL_MODEL_FEATURES = ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES


def calculate_profit_margin(df: pd.DataFrame) -> pd.Series:
    """
    Calculates profit margin percentage: (Profit / Sales) * 100.
    
    CRITICAL ARCHITECTURAL NOTE:
    This metric is computed strictly for historical dataset reporting, EDA tracking,
    and post-prediction evaluation in the BI dashboard. It is NEVER supplied as an
    input feature (X) during regression training to prevent 100% target leakage!
    """
    if "Profit" in df.columns and "Sales" in df.columns:
        return (df["Profit"] / df["Sales"]) * 100.0
    return pd.Series(index=df.index, dtype=float)


def engineer_features(df: pd.DataFrame, include_target_analytics: bool = True) -> pd.DataFrame:
    """
    Applies deterministic domain feature engineering to a given DataFrame.

    Features Created:
    -----------------
    1. High_Discount_Flag : 1 if Discount_pct >= 20% else 0 (strong return & margin drop threshold)
    2. Late_Delivery_Flag : 1 if Delivery_Days > 5 else 0 (shipping SLA breach threshold)
    3. Discount_Bucket    : Categorical bins: Low (0-10%), Medium (11-20%), High (21-30%), Very High (>30%)
    4. Sales_Bucket       : Categorical bins: Low (<30k), Medium (30k-60k), High (60k-90k), Very High (>90k)
    5. Profit_Margin      : (Profit / Sales) * 100 (for analytics/reporting if Profit present)
    6. Region_Category    : String interaction: Region + "_" + Category (e.g. South_Furniture)
    7. Delivery_Category  : String interaction: Category + "_" + Late/OnTime delivery flag
    8. Low_Rating_Flag    : 1 if Customer_Rating < 3.5 else 0 (critical satisfaction drop)
    9. High_Risk_Combination_Flag : 1 if High_Discount_Flag AND Late_Delivery_Flag else 0
    10. Effective_Sales   : Sales * (1 - Discount_pct / 100)
    11. Discount_Amount   : Sales * (Discount_pct / 100)
    12. Rating_Delivery_Ratio : Customer_Rating / (Delivery_Days + 1)
    """
    data = df.copy()

    # 1. High_Discount_Flag (EDA revealed returns surge from 12.1% to 44.1% above 20% discount)
    data["High_Discount_Flag"] = (data["Discount_pct"] >= 20).astype(int)

    # 2. Late_Delivery_Flag (Mean delivery is 5.27 days; deliveries > 5 days double return risk)
    data["Late_Delivery_Flag"] = (data["Delivery_Days"] > 5).astype(int)

    # 3. Discount_Bucket
    discount_bins = [-np.inf, 10, 20, 30, np.inf]
    discount_labels = ["Low (0-10%)", "Medium (11-20%)", "High (21-30%)", "Very High (>30%)"]
    data["Discount_Bucket"] = pd.cut(
        data["Discount_pct"], bins=discount_bins, labels=discount_labels
    ).astype(str)

    # 4. Sales_Bucket
    sales_bins = [-np.inf, 30000, 60000, 90000, np.inf]
    sales_labels = ["Low (<30k)", "Medium (30k-60k)", "High (60k-90k)", "Very High (>90k)"]
    data["Sales_Bucket"] = pd.cut(
        data["Sales"], bins=sales_bins, labels=sales_labels
    ).astype(str)

    # 5. Profit_Margin (Computed for reporting/interim data if Profit column is present)
    if include_target_analytics and "Profit" in data.columns:
        data["Profit_Margin"] = calculate_profit_margin(data)

    # 6. Region_Category Combination
    data["Region_Category"] = data["Region"].astype(str) + "_" + data["Category"].astype(str)

    # 7. Delivery_Category Interaction
    delivery_status = np.where(data["Delivery_Days"] > 5, "Late", "Normal")
    data["Delivery_Category"] = data["Category"].astype(str) + "_" + delivery_status

    # 8. Low_Rating_Flag (Rating < 3.5 exhibits 69.3% return rate)
    data["Low_Rating_Flag"] = (data["Customer_Rating"] < 3.5).astype(int)

    # 9. High_Risk_Combination_Flag (Concurrent high discount and late delivery)
    data["High_Risk_Combination_Flag"] = (
        (data["High_Discount_Flag"] == 1) & (data["Late_Delivery_Flag"] == 1)
    ).astype(int)

    # 10. Effective_Sales (Net realization post-discount)
    data["Effective_Sales"] = data["Sales"] * (1.0 - data["Discount_pct"] / 100.0)

    # 11. Discount_Amount (Total monetary discount conceded)
    data["Discount_Amount"] = data["Sales"] * (data["Discount_pct"] / 100.0)

    # 12. Rating_Delivery_Ratio (Efficiency index)
    data["Rating_Delivery_Ratio"] = data["Customer_Rating"] / (data["Delivery_Days"] + 1.0)

    return data


class NovaCartFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible Transformer for NovaCart Feature Engineering.
    Can be seamlessly serialized inside an sklearn Pipeline to eliminate
    training-serving skew in production or Streamlit deployment.
    """

    def __init__(self, include_target_analytics: bool = False):
        self.include_target_analytics = include_target_analytics

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=RAW_INPUT_FEATURES)
        return engineer_features(X, include_target_analytics=self.include_target_analytics)


def run_feature_engineering_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads train/test splits, applies feature engineering, and exports processed datasets.
    """
    train_path = DATA_DIR / "train_orders.csv"
    test_path = DATA_DIR / "test_orders.csv"

    if not train_path.exists() or not test_path.exists():
        logger.info("Processed train/test files not found. Triggering preprocessing pipeline first...")
        from src.preprocessing import run_preprocessing_pipeline
        run_preprocessing_pipeline()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Engineering features for Train set ({len(train_df)} rows)...")
    fe_train_df = engineer_features(train_df, include_target_analytics=True)

    logger.info(f"Engineering features for Test set ({len(test_df)} rows)...")
    fe_test_df = engineer_features(test_df, include_target_analytics=True)

    # Save processed feature-engineered splits
    fe_train_path = DATA_DIR / "train_featured.csv"
    fe_test_path = DATA_DIR / "test_featured.csv"

    fe_train_df.to_csv(fe_train_path, index=False)
    fe_test_df.to_csv(fe_test_path, index=False)

    logger.info(f"Saved featured train set ({fe_train_df.shape}) to: {fe_train_path}")
    logger.info(f"Saved featured test set  ({fe_test_df.shape}) to: {fe_test_path}")

    # Also save full combined processed data for reporting and EDA
    cleaned_path = DATA_DIR / "cleaned_orders.csv"
    if cleaned_path.exists():
        full_df = pd.read_csv(cleaned_path)
        fe_full_df = engineer_features(full_df, include_target_analytics=True)
        full_fe_path = DATA_DIR / "processed_orders.csv"
        fe_full_df.to_csv(full_fe_path, index=False)
        logger.info(f"Saved complete processed dataset ({fe_full_df.shape}) to: {full_fe_path}")

    return fe_train_df, fe_test_df


if __name__ == "__main__":
    logger.info("Executing feature_engineering.py standalone verification...")
    run_feature_engineering_pipeline()
    logger.info("Feature engineering completed successfully.")
