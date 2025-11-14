"""
vehicle_eda_cleaning.py

End-to-end loading + cleaning + feature-engineering for the
'Vehicle Sales Data' Kaggle dataset.

Focus:
- Standardize and clean key columns
- Remove/flag bad/missing values
- Create modeling-ready core dataset for:
  * Price prediction using brand and model year
  * Price vs odometer + condition
  * Brand/model-year/state volume patterns
"""

import argparse
from typing import Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 1. Loading & basic overview
# -----------------------------

def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load the raw vehicle sales CSV into a DataFrame.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


def basic_overview(df: pd.DataFrame) -> None:
    """
    Print basic info, missingness, and cardinality.
    """
    print("\n=== DataFrame info ===")
    print(df.info())

    print("\n=== Missing values per column ===")
    print(df.isna().sum().sort_values(ascending=False))

    print("\n=== Unique values per column ===")
    print(df.nunique().sort_values(ascending=False))


# -----------------------------
# 2. Helper functions
# -----------------------------

def _standardize_str_series(s: pd.Series) -> pd.Series:
    """
    Lowercase, strip whitespace, and replace empty strings with NaN.
    """
    s = s.astype("string")
    s = s.str.strip().str.lower()
    s = s.replace({"": pd.NA})
    return s


def _map_body_type(body: pd.Series) -> pd.Series:
    """
    Map raw 'body' strings to a smaller, consistent set.
    """
    body = _standardize_str_series(body)

    mapping = {
        "suv": "suv",
        "sport utility": "suv",
        "crossover": "suv",
        "pickup": "pickup",
        "pickup truck": "pickup",
        "truck": "pickup",
        "sedan": "sedan",
        "saloon": "sedan",
        "coupe": "coupe",
        "coupe 2d": "coupe",
        "hatchback": "hatchback",
        "wagon": "wagon",
        "van": "van",
        "minivan": "van",
        "convertible": "convertible"
    }

    return body.map(mapping).fillna("other")


def _map_transmission(trans: pd.Series) -> pd.Series:
    """
    Map raw 'transmission' strings into {automatic, manual, other, unknown}.
    """
    trans = _standardize_str_series(trans)

    automatic_keywords = ["auto", "automatic"]
    manual_keywords = ["manual", "man", "stick"]

    def classify(val: str) -> str:
        if val is None or pd.isna(val):
            return "unknown"
        for kw in automatic_keywords:
            if kw in val:
                return "automatic"
        for kw in manual_keywords:
            if kw in val:
                return "manual"
        return "other"

    return trans.apply(classify)


def _map_color(color: pd.Series) -> pd.Series:
    """
    Map raw exterior color to a small set of color groups.
    """
    color = _standardize_str_series(color)

    # Handle some common multi-color or weird tokens
    color = color.str.replace("/", " ", regex=False)
    color = color.str.replace("-", " ", regex=False)

    def simplify(val: str) -> str:
        if val is None or pd.isna(val):
            return "unknown"

        if "white" in val:
            return "white"
        if "black" in val:
            return "black"
        if "silver" in val or "sliver" in val:
            return "silver"
        if "grey" in val or "gray" in val:
            return "gray"
        if "blue" in val:
            return "blue"
        if "red" in val or "maroon" in val or "burgundy" in val:
            return "red"
        if "green" in val:
            return "green"
        if "gold" in val or "champagne" in val or "beige" in val or "tan" in val:
            return "gold_beige"
        if "brown" in val or "bronze" in val:
            return "brown"
        if "yellow" in val:
            return "yellow"
        if "orange" in val:
            return "orange"
        if "purple" in val:
            return "purple"

        # Everything else
        return "other"

    return color.apply(simplify)


def _standardize_state(state: pd.Series) -> pd.Series:
    """
    Clean up US state codes: uppercase, strip, map invalid to 'unknown'.
    """
    valid_state_codes = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC"
    }

    state = state.astype("string").str.strip().str.upper()
    return state.where(state.isin(valid_state_codes), other="UNKNOWN")


def _clean_condition(cond: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Clean and rescale 'condition'.

    Strategy:
    - Convert to numeric, coercing errors to NaN.
    - Handle the dual-scale issue:
        * If cond <= 5, treat as 1–5 and scale to 10–50 by *10.
        * Otherwise, assume already on 10–50 scale.
    - Create a 0–1 standardized version for modeling.
    """
    cond_num = pd.to_numeric(cond, errors="coerce")

    # Rescale the small scale (1–5) to the larger 10–50 style
    mask_small_scale = cond_num <= 5
    cond_scaled = cond_num.copy()
    cond_scaled[mask_small_scale] = cond_scaled[mask_small_scale] * 10

    # Now standardize to 0–1 based on observed min/max (ignoring NaN)
    min_val = cond_scaled.min(skipna=True)
    max_val = cond_scaled.max(skipna=True)
    cond_std = (cond_scaled - min_val) / (max_val - min_val)

    return cond_scaled, cond_std


def _clean_numeric_outliers(
    s: pd.Series,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
    non_positive_to_nan: bool = False
) -> pd.Series:
    """
    Clip or drop extreme outliers in a numeric Series using quantiles.

    - Convert to numeric and coerce errors to NaN.
    - If non_positive_to_nan=True, values <= 0 become NaN.
    - Values outside [q_low, q_high] are set to NaN (to later be dropped).
    """
    s_num = pd.to_numeric(s, errors="coerce")

    if non_positive_to_nan:
        s_num = s_num.mask(s_num <= 0, other=np.nan)

    q_low = s_num.quantile(lower_quantile)
    q_high = s_num.quantile(upper_quantile)

    s_num = s_num.mask((s_num < q_low) | (s_num > q_high), other=np.nan)

    return s_num


# -----------------------------
# 3. Main cleaning pipeline
# -----------------------------

def clean_vehicle_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and feature-engineer the vehicle dataset.

    Returns
    -------
    df_clean : DataFrame
        Cleaned dataset (still fairly wide; some non-core columns kept).
    df_model_core : DataFrame
        Narrowed modeling-ready dataset with key engineered features.
    """
    df = df_raw.copy()

    # --- 3.1 Basic string standardization for key categoricals ---
    if "make" in df.columns:
        df["make"] = _standardize_str_series(df["make"])

    for col in ["model", "trim", "seller"]:
        if col in df.columns:
            df[col] = _standardize_str_series(df[col])

    if "body" in df.columns:
        df["body_group"] = _map_body_type(df["body"])

    if "transmission" in df.columns:
        df["transmission_group"] = _map_transmission(df["transmission"])

    if "color" in df.columns:
        df["color_group"] = _map_color(df["color"])
    elif "colour" in df.columns:
        df["color_group"] = _map_color(df["colour"])

    if "interior" in df.columns:
        df["interior_clean"] = _standardize_str_series(df["interior"])

    if "state" in df.columns:
        df["state_clean"] = _standardize_state(df["state"])

    # --- 3.2 Parse sale date and create time features (context only) ---
    if "saledate" in df.columns:
        df["saledate_parsed"] = pd.to_datetime(
            df["saledate"],
            errors="coerce",
            utc=True
        )
        # Drop rows where we couldn't parse a date
        before = df.shape[0]
        df = df[df["saledate_parsed"].notna()]
        after = df.shape[0]
        print(f"Dropped {before - after} rows with unparseable saledate.")

        df["sale_year"] = df["saledate_parsed"].dt.year
        df["sale_month"] = df["saledate_parsed"].dt.month
        df["sale_quarter"] = df["saledate_parsed"].dt.quarter

    # --- 3.3 Clean year and create vehicle age ---
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = np.nan

    # NOTE: we no longer filter by sale_year window for modeling,
    # since all sales are in 2014–2015 and we focus on model year.

    # Vehicle age at sale: sale_year - manufacture year (still useful as a feature)
    if "sale_year" in df.columns:
        df["vehicle_age_at_sale"] = df["sale_year"] - df["year"]
    else:
        df["vehicle_age_at_sale"] = np.nan

    # --- 3.4 Clean condition (dual scale) ---
    if "condition" in df.columns:
        df["condition_scaled"], df["condition_std"] = _clean_condition(df["condition"])
    else:
        df["condition_scaled"] = np.nan
        df["condition_std"] = np.nan

    # --- 3.5 Clean numeric fields: odometer, mmr, sellingprice ---
    if "odometer" in df.columns:
        df["odometer_clean"] = _clean_numeric_outliers(
            df["odometer"], lower_quantile=0.001, upper_quantile=0.999
        )
        df["log_odometer"] = np.log1p(df["odometer_clean"])
    else:
        df["odometer_clean"] = np.nan
        df["log_odometer"] = np.nan

    if "mmr" in df.columns:
        df["mmr_clean"] = _clean_numeric_outliers(
            df["mmr"], lower_quantile=0.001, upper_quantile=0.999,
            non_positive_to_nan=True
        )
    else:
        df["mmr_clean"] = np.nan

    if "sellingprice" in df.columns:
        df["sellingprice_clean"] = _clean_numeric_outliers(
            df["sellingprice"], lower_quantile=0.001, upper_quantile=0.999,
            non_positive_to_nan=True
        )
        df["log_sellingprice"] = np.log1p(df["sellingprice_clean"])
    else:
        df["sellingprice_clean"] = np.nan
        df["log_sellingprice"] = np.nan

    # --- 3.6 Drop exact duplicates and VIN duplicates ---
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Dropped {before - after} exact duplicate rows.")

    if "vin" in df.columns and "saledate_parsed" in df.columns:
        before = df.shape[0]
        df = df.drop_duplicates(subset=["vin", "saledate_parsed"])
        after = df.shape[0]
        print(f"Dropped {before - after} duplicate (vin, saledate) rows.")

    # --- 3.7 Precompute model-year volume features (for RQ3) ---
    # Use cleaned df, but don't require odometer/condition/etc for counts
    vol_base = df[["make", "year", "state_clean"]].dropna(
        subset=["make", "year", "state_clean"]
    )

    model_year_counts = (
        vol_base.groupby("year")
        .size()
        .rename("total_sold_by_model_year")
    )

    make_year_counts = (
        vol_base.groupby(["make", "year"])
        .size()
        .rename("total_sold_by_make_model_year")
    )

    make_year_state_counts = (
        vol_base.groupby(["make", "year", "state_clean"])
        .size()
        .rename("total_sold_by_make_model_year_state")
    )

    # --- 3.8 Build core modeling dataset ---
    core_cols = [
        "make",
        "year",
        "sale_year",
        "sale_month",
        "sale_quarter",
        "vehicle_age_at_sale",
        "body_group",
        "transmission_group",
        "color_group",
        "state_clean",
        "condition_scaled",
        "condition_std",
        "odometer_clean",
        "log_odometer",
        "mmr_clean",
        "sellingprice_clean",
        "log_sellingprice",
        # volume features
        "total_sold_by_model_year",
        "total_sold_by_make_model_year",
        "total_sold_by_make_model_year_state",
    ]

    # Ensure all core columns exist in df
    for c in core_cols:
        if c not in df.columns:
            df[c] = np.nan

    df_model_core = df[core_cols].copy()

    # Merge volume features from groupby results
    df_model_core = df_model_core.merge(
        model_year_counts, on="year", how="left", suffixes=("", "_from_year_counts")
    )
    df_model_core["total_sold_by_model_year"] = df_model_core[
        "total_sold_by_model_year_from_year_counts"
    ].fillna(df_model_core["total_sold_by_model_year"])
    df_model_core.drop(columns=["total_sold_by_model_year_from_year_counts"], inplace=True)

    df_model_core = df_model_core.merge(
        make_year_counts, on=["make", "year"], how="left", suffixes=("", "_from_make_year")
    )
    df_model_core["total_sold_by_make_model_year"] = df_model_core[
        "total_sold_by_make_model_year_from_make_year"
    ].fillna(df_model_core["total_sold_by_make_model_year"])
    df_model_core.drop(columns=["total_sold_by_make_model_year_from_make_year"], inplace=True)

    df_model_core = df_model_core.merge(
        make_year_state_counts,
        on=["make", "year", "state_clean"],
        how="left",
        suffixes=("", "_from_make_year_state"),
    )
    df_model_core["total_sold_by_make_model_year_state"] = df_model_core[
        "total_sold_by_make_model_year_state_from_make_year_state"
    ].fillna(df_model_core["total_sold_by_make_model_year_state"])
    df_model_core.drop(
        columns=["total_sold_by_make_model_year_state_from_make_year_state"],
        inplace=True,
    )

    # --- 3.9 Drop rows missing any absolutely essential fields for your analyses ---
    # Focus on model year, brand, state, price, odometer, condition, color.
    essential = [
        "make",
        "year",
        "state_clean",
        "sellingprice_clean",
        "odometer_clean",
        "condition_std",
        "color_group",
    ]
    before = df_model_core.shape[0]
    df_model_core = df_model_core.dropna(subset=essential)
    after = df_model_core.shape[0]
    print(f"Dropped {before - after} rows from df_model_core due to missing essentials.")

    print("\n=== Final shapes ===")
    print(f"df_clean      : {df.shape}")
    print(f"df_model_core : {df_model_core.shape}")

    return df, df_model_core


# -----------------------------
# 4. CLI entry point (CSV version)
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean vehicle sales dataset.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the raw vehicle sales CSV file."
    )
    parser.add_argument(
        "--output_clean",
        type=str,
        default="data/cleaned/vehicle_clean_full.csv",
        help="Where to save the cleaned full dataset (CSV)."
    )
    parser.add_argument(
        "--output_core",
        type=str,
        default="data/cleaned/vehicle_model_core.csv",
        help="Where to save the modeling-ready core dataset (CSV)."
    )

    args = parser.parse_args()

    df_raw = load_raw_data(args.csv_path)
    basic_overview(df_raw)

    df_clean, df_model_core = clean_vehicle_data(df_raw)

    # Make sure the output directories exist
    import os
    os.makedirs(os.path.dirname(args.output_clean), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_core), exist_ok=True)

    # SAVE AS CSV
    df_clean.to_csv(args.output_clean, index=False)
    df_model_core.to_csv(args.output_core, index=False)

    print(f"\nSaved cleaned full dataset to: {args.output_clean}")
    print(f"Saved modeling core dataset to: {args.output_core}")


if __name__ == "__main__":
    main()
