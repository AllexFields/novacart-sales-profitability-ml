"""
NovaCart AI Business Intelligence System - Data Preprocessing Pipeline
Handles raw dataset ingestion, column normalization, schema validation,
type casting, missing-value imputation, target encoding, and stratified splitting.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Support direct execution and package-level imports
try:
    from src.utils import DATA_DIR, RANDOM_STATE, setup_logger
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.utils import DATA_DIR, RANDOM_STATE, setup_logger

logger = setup_logger("preprocessing")

# Expected raw columns and data types
EXPECTED_COLUMNS = [
    "Order_ID",
    "Order_Date",
    "Region",
    "Category",
    "Sales",
    "Discount_pct",
    "Profit",
    "Delivery_Days",
    "Customer_Rating",
    "Returned",
]

NUMERIC_COLUMNS = ["Sales", "Discount_pct", "Profit", "Delivery_Days", "Customer_Rating"]
CATEGORICAL_COLUMNS = ["Region", "Category"]


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the raw orders dataset from an Excel spreadsheet or CSV file.

    Parameters
    ----------
    filepath : Optional[Path]
        Custom path to the raw dataset file. If None, defaults to `data/NovaCart_Orders.xlsx`.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    if filepath is None:
        filepath = DATA_DIR / "NovaCart_Orders.xlsx"
        if not filepath.exists():
            # Fallback to alternative naming if present in workspace root
            fallback = Path("NovaCart_Data_Analysis_Case_Study (2).xlsx")
            if fallback.exists():
                filepath = fallback

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {filepath.resolve()}")

    logger.info(f"Loading raw data from: {filepath.resolve()}")

    if filepath.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath, sheet_name=0)
    elif filepath.suffix.lower() == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Expected .xlsx or .csv.")

    logger.info(f"Raw data successfully ingested. Initial shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, standardizes column naming, strips whitespace,
    validates numeric constraints, and encodes the binary classification target.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned and validated DataFrame with `Returned_Binary` target column added.
    """
    data = df.copy()

    # 1. Standardize column names (strip whitespace, normalize casing)
    col_mapping = {}
    for col in data.columns:
        clean_col = col.strip()
        # Ensure Discount_pct matches standardized casing
        if clean_col.lower() in ["discount_pct", "discount_percentage", "discount"]:
            col_mapping[col] = "Discount_pct"
        elif clean_col.lower() == "customer_rating":
            col_mapping[col] = "Customer_Rating"
        elif clean_col.lower() == "delivery_days":
            col_mapping[col] = "Delivery_Days"
        else:
            col_mapping[col] = clean_col
    data = data.rename(columns=col_mapping)

    # 2. Check for presence of required columns
    missing_cols = set(EXPECTED_COLUMNS) - set(data.columns)
    if missing_cols:
        raise KeyError(f"Dataset is missing required columns: {missing_cols}")

    # 3. Clean string / categorical fields
    for cat_col in CATEGORICAL_COLUMNS + ["Returned"]:
        if cat_col in data.columns:
            data[cat_col] = data[cat_col].astype(str).str.strip().str.title()

    # 4. Standardize Order_Date to datetime
    if "Order_Date" in data.columns:
        data["Order_Date"] = pd.to_datetime(data["Order_Date"], dayfirst=True, errors="coerce")

    # 5. Type-cast numeric columns and handle missing values
    for num_col in NUMERIC_COLUMNS:
        data[num_col] = pd.to_numeric(data[num_col], errors="coerce")
        if data[num_col].isnull().any():
            median_val = data[num_col].median()
            logger.warning(f"Imputing {data[num_col].isnull().sum()} missing values in '{num_col}' with median: {median_val}")
            data[num_col] = data[num_col].fillna(median_val)

    # 6. Encode Target Variable for Classification
    # 'Yes' -> 1 (Returned), 'No' -> 0 (Retained)
    data["Returned_Binary"] = data["Returned"].map({"Yes": 1, "No": 0})
    if data["Returned_Binary"].isnull().any():
        invalid_returned = data[data["Returned_Binary"].isnull()]["Returned"].unique()
        raise ValueError(f"Unrecognized target values in 'Returned' column: {invalid_returned}")
    data["Returned_Binary"] = data["Returned_Binary"].astype(int)

    # 7. Validate value ranges and business logic sanity
    assert (data["Sales"] > 0).all(), "Sales must be strictly positive."
    assert (data["Discount_pct"] >= 0).all() and (data["Discount_pct"] <= 100).all(), "Discount_pct must be in [0, 100]."
    assert (data["Delivery_Days"] >= 0).all(), "Delivery_Days cannot be negative."
    assert (data["Customer_Rating"] >= 1.0).all() and (data["Customer_Rating"] <= 5.0).all(), "Customer_Rating must be between 1.0 and 5.0."

    logger.info(f"Data cleaning and validation complete. Final cleaned shape: {data.shape}")
    logger.info(f"Target distribution ('Returned'):\n{data['Returned'].value_counts(normalize=True).to_dict()}")

    return data


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    stratify_col: Optional[str] = "Returned_Binary"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train/test split to preserve return-rate distribution across folds.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned orders dataset.
    test_size : float
        Proportion of dataset to allocate to the holdout test set (default: 0.20).
    random_state : int
        Seed for deterministic splitting.
    stratify_col : Optional[str]
        Column name used for stratification (default: 'Returned_Binary').

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    logger.info(f"Dataset split complete: Train shape = {train_df.shape}, Test shape = {test_df.shape}")
    if stratify_col and stratify_col in df.columns:
        logger.info(f"Train Return Rate: {train_df[stratify_col].mean():.2%}")
        logger.info(f"Test Return Rate:  {test_df[stratify_col].mean():.2%}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_preprocessing_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the end-to-end preprocessing workflow and saves the processed artifacts.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Cleaned train and test DataFrames.
    """
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)

    # Save complete cleaned data
    cleaned_path = DATA_DIR / "cleaned_orders.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    logger.info(f"Saved cleaned dataset to: {cleaned_path}")

    # Split into train and test
    train_df, test_df = split_data(cleaned_df)

    train_path = DATA_DIR / "train_orders.csv"
    test_path = DATA_DIR / "test_orders.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved train split ({len(train_df)} rows) to: {train_path}")
    logger.info(f"Saved test split  ({len(test_df)} rows) to: {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    logger.info("Executing preprocessing.py standalone verification...")
    run_preprocessing_pipeline()
    logger.info("Data preprocessing completed successfully.")
