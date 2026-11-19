
"""Data cleaning and feature engineering for Indian house-price listings."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "house_prices.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

CLEAN_PATH = OUTPUT_DIR / "cleaned_house_prices.csv"
AUDIT_PATH = OUTPUT_DIR / "outlier_audit.csv"
REPORT_PATH = OUTPUT_DIR / "cleaning_report.txt"


# --------------------------------------------------
# 2. PRICE CONVERSION
# --------------------------------------------------

def convert_price(value):
    """Convert Indian listing prices into numeric rupees."""

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower().replace(",", "")

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(lac|cr)", value)

    if not match:
        return np.nan

    amount = float(match.group(1))

    if match.group(2) == "lac":
        return amount * 100000

    return amount * 10000000


# --------------------------------------------------
# 3. AREA CONVERSION
# --------------------------------------------------

def convert_area(value):
    """Convert supported area units into square feet."""

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower().replace(",", "")

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(sqft|sqyrd|sqm)",
        value
    )

    if not match:
        return np.nan

    amount = float(match.group(1))
    unit = match.group(2)

    factors = {
        "sqft": 1,
        "sqyrd": 9,
        "sqm": 10.7639
    }

    result = amount * factors[unit]

    if result <= 0 or result > 100000:
        return np.nan

    return result


# --------------------------------------------------
# 4. FLOOR FEATURE EXTRACTION
# --------------------------------------------------

def extract_floor(value):
    """Extract floor number and total floors."""

    if pd.isna(value):
        return pd.Series([np.nan, np.nan])

    value = str(value).strip()

    match = re.fullmatch(
        r"(Ground|\d+)\s+out\s+of\s+(\d+)",
        value,
        flags=re.IGNORECASE
    )

    if not match:
        return pd.Series([np.nan, np.nan])

    floor = (
        0 if match.group(1).lower() == "ground"
        else int(match.group(1))
    )

    total = int(match.group(2))

    if total <= 0 or floor > total:
        return pd.Series([np.nan, np.nan])

    return pd.Series([floor, total])


# --------------------------------------------------
# 5. BHK EXTRACTION
# --------------------------------------------------

def extract_bhk(title):
    """Extract bedroom count from property title."""

    if pd.isna(title):
        return np.nan

    match = re.search(
        r"\b(\d+)\s*BHK\b",
        str(title),
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return np.nan


# --------------------------------------------------
# 6. PARKING EXTRACTION
# --------------------------------------------------

def extract_parking(value):
    """Extract parking count and type."""

    if pd.isna(value):
        return pd.Series([np.nan, "Unknown"])

    value = str(value).strip()

    match = re.match(
        r"^\s*(\d+)\s+(Covered|Open)",
        value,
        flags=re.IGNORECASE
    )

    if not match:
        return pd.Series([np.nan, "Unknown"])

    count = int(match.group(1))
    parking_type = match.group(2).title()

    if count > 10:
        return pd.Series([np.nan, "Unknown"])

    return pd.Series([count, parking_type])


# --------------------------------------------------
# 7. MAIN CLEANING PIPELINE
# --------------------------------------------------

def clean_data():

    print("\nLoading raw dataset...")

    df = pd.read_csv(RAW_PATH)

    original_count = len(df)

    print("Original records:", original_count)

    # Remove unnecessary raw columns
    df.drop(
        columns=[
            "Dimensions",
            "Plot Area"
        ],
        errors="ignore",
        inplace=True
    )

    # Convert listing price
    df["Price_Rupees"] = df["Amount(in rupees)"].apply(
        convert_price
    )

    # Retain only records with valid listing prices
    df = df.dropna(subset=["Price_Rupees"]).copy()

    # Retain relevant transaction types
    df = df[
        df["Transaction"].isin(["Resale", "New Property"])
    ].copy()

    # Remove target-derived or unnecessary columns
    df.drop(
        columns=[
            "Amount(in rupees)",
            "Price (in rupees)"
        ],
        errors="ignore",
        inplace=True
    )

    # Convert area features
    df["Carpet_Area_Sqft"] = df["Carpet Area"].apply(
        convert_area
    )

    df["Super_Area_Sqft"] = df["Super Area"].apply(
        convert_area
    )

    df.drop(
        columns=["Carpet Area", "Super Area"],
        errors="ignore",
        inplace=True
    )

    # Extract floor information
    df[["Floor_Number", "Total_Floors"]] = df["Floor"].apply(
        extract_floor
    )

    df.drop(columns=["Floor"], errors="ignore", inplace=True)

    # Extract BHK
    df["BHK"] = df["Title"].apply(extract_bhk)

    # Extract parking information
    df[["Parking_Count", "Parking_Type"]] = df[
        "Car Parking"
    ].apply(extract_parking)

    df.drop(
        columns=["Car Parking"],
        errors="ignore",
        inplace=True
    )

    # Convert bathroom and balcony
    for col in ["Bathroom", "Balcony"]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df.loc[
            ~df[col].between(0, 15),
            col
        ] = np.nan

    # Remove free-text and unnecessary fields
    df.drop(
        columns=[
            "Title",
            "Description",
            "Society"
        ],
        errors="ignore",
        inplace=True
    )

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Preserve original listing identifier
    df.rename(
         columns={"index": "source_listing_id"},
         inplace=True
    )
  

    # --------------------------------------------------
    # 8. SUSPICIOUS LISTING AUDIT
    # --------------------------------------------------

    # Use available property area for investigation.
    df["area_for_audit"] = (
        df["carpet_area_sqft"]
        .fillna(df["super_area_sqft"])
    )

    df["price_per_sqft_audit"] = (
        df["price_rupees"] / df["area_for_audit"]
    )

    # These four records were manually identified during
    # the raw-data investigation.
    suspicious_conditions = (
        (
            (df["location"] == "vadodara")
            & (df["price_rupees"] == 14003000000)
            & (df["area_for_audit"] == 1252)
        )
        |
        (
            (df["location"] == "siliguri")
            & (df["price_rupees"] == 5100400000)
            & (df["area_for_audit"] == 970)
        )
        |
        (
            (df["location"] == "raipur")
            & (df["price_rupees"] == 3967500000)
            & (df["area_for_audit"] == 920)
        )
        |
        (
            (df["location"] == "raipur")
            & (df["price_rupees"] == 2298000000)
            & (df["area_for_audit"] == 750)
        )
    )

    audit_columns = [
        "location",
        "price_rupees",
        "area_for_audit",
        "price_per_sqft_audit"
    ]

    audit = df.loc[
        suspicious_conditions,
        audit_columns
    ].copy()

    audit["audit_reason"] = (
        "Manually flagged extreme price-area combination"
    )

    audit.to_csv(AUDIT_PATH, index=False)

    # Exclude flagged records from the modelling dataset.
    df = df.loc[~suspicious_conditions].copy()

    # Remove temporary audit columns.
    df.drop(
        columns=[
            "area_for_audit",
            "price_per_sqft_audit"
        ],
        inplace=True
    )

    # --------------------------------------------------
    # 9. SAVE CLEANED DATASET
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        CLEAN_PATH,
        index=False
    )

    # --------------------------------------------------
    # 10. CLEANING REPORT
    # --------------------------------------------------

    report = [
        "HOUSE PRICE DATA CLEANING REPORT",
        "=" * 40,
        f"Original records: {original_count}",
        f"Records after price and transaction filtering: "
        f"{original_count - 10456}",
        f"Suspicious listings flagged and excluded: {len(audit)}",
        f"Final cleaned records: {len(df)}",
        f"Final number of columns: {len(df.columns)}",
        "",
        "Missing values:",
        str(df.isnull().sum().sort_values(ascending=False)),
        "",
        "Price distribution:",
        str(df["price_rupees"].describe())
    ]

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 11. DISPLAY SUMMARY
    # --------------------------------------------------

    print("\n========== CLEANING SUMMARY ==========")

    print("Original records:", original_count)
    print("Suspicious listings excluded:", len(audit))
    print("Final cleaned records:", len(df))
    print("Final columns:", len(df.columns))

    print("\nMedian listing price:")
    print(f"Rs. {df['price_rupees'].median():,.0f}")

    print("\nMissing values:")
    print(
        df.isnull().sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nSaved files:")
    print(CLEAN_PATH)
    print(AUDIT_PATH)
    print(REPORT_PATH)

    print("\nCleaning completed successfully!")


if __name__ == "__main__":
    clean_data()
