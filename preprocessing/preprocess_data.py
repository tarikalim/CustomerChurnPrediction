import os
import pandas as pd
from config import INPUT_EXCEL_PATH, OUTPUT_CSV_PATH


def load_data(path: str = INPUT_EXCEL_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns"""
    to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude',
        'Churn Label', 'Churn Reason', 'Churn Score', 'CLTV'
    ]
    existing = [c for c in to_drop if c in df.columns]
    return df.drop(columns=existing)


def ensure_numeric_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Total Charges to numeric and fill missing."""
    df.loc[:, 'Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    median_val = df['Total Charges'].median()
    df.loc[:, 'Total Charges'] = df['Total Charges'].fillna(median_val)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for modeling."""
    return df.rename(columns={
        'Churn Value': 'Churn',
        'Tenure Months': 'tenure',
        'Monthly Charges': 'MonthlyCharges',
        'Total Charges': 'TotalCharges'
    })


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No columns to 1/0 for binary features and label."""
    binary_cols = ['Senior Citizen', 'Partner', 'Dependents',
                   'Phone Service', 'Paperless Billing']
    for col in binary_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].map({'Yes': 1, 'No': 0})
    df.loc[:, 'Churn'] = df['Churn'].astype(int)
    return df


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical service and contract columns."""
    cat_cols = [
        'Gender', 'Multiple Lines', 'Internet Service', 'Online Security',
        'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
        'Streaming Movies', 'Contract', 'Payment Method'
    ]
    existing = [c for c in cat_cols if c in df.columns]
    return pd.get_dummies(df, columns=existing, drop_first=True)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run preprocessing steps."""
    df = clean_columns(df)
    df = ensure_numeric_total_charges(df)
    df = rename_columns(df)
    df = encode_binary(df)
    df = one_hot(df)
    return df


def preprocess_file(
        input_path: str = INPUT_EXCEL_PATH,
        output_path: str = OUTPUT_CSV_PATH
) -> pd.DataFrame:
    df = load_data(input_path)
    df_clean = preprocess_dataframe(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    return df_clean
