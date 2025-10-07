from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


# Configure a basic logger for this utility module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _project_file(default_filename: str, explicit_path: Optional[str] = None) -> Path:
    """Resolve a file path relative to this module unless an explicit path is provided.

    Args:
        default_filename: The default CSV filename to load from the same directory as this file.
        explicit_path: Optional explicit path to the CSV file; can be absolute or relative.

    Returns:
        A resolved Path to the target file.
    """
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return (Path(__file__).parent / default_filename).resolve()


def _coerce_quantity_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a `Quantity` column exists; fall back to `Qty` if present.

    - If `Quantity` is missing but `Qty` exists, rename `Qty` -> `Quantity`.
    - Attempts to coerce the `Quantity` column to numeric (NaN on failures).
    """
    if "Quantity" not in df.columns and "Qty" in df.columns:
        df = df.rename(columns={"Qty": "Quantity"})
        logger.info("Renamed 'Qty' column to 'Quantity'.")

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    else:
        logger.warning("No 'Quantity' or 'Qty' column found.")
    return df


def _try_parse_dates(df: pd.DataFrame, candidate_columns: Tuple[str, ...]) -> pd.DataFrame:
    """Parse date-like columns in-place when present.

    For each column name in candidate_columns that exists in the DataFrame, attempt
    to parse it to datetime with errors coerced to NaT.
    """
    for col in candidate_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            logger.info("Parsed column '%s' as datetime.", col)
    return df


def load_sales_csv(path: Optional[str] = None) -> pd.DataFrame:
    """Load sales CSV safely with date parsing and quantity normalization.

    Behavior:
    - Resolves the CSV path to `sales.csv` in the same folder as this file by default.
    - Parses common date columns if present: `Date`, `OrderDate`, `TransactionDate`, `SaleDate`.
    - Ensures a `Quantity` column exists (renames `Qty` -> `Quantity` if needed) and coerces to numeric.

    Args:
        path: Optional explicit path to the sales CSV.

    Returns:
        A pandas DataFrame.
    """
    csv_path = _project_file("sales.csv", path)
    logger.info("Loading sales CSV from: %s", csv_path)
    df = pd.read_csv(csv_path)

    df = _try_parse_dates(df, ("Date", "OrderDate", "TransactionDate", "SaleDate"))
    df = _coerce_quantity_column(df)
    return df


def load_stocks_csv(path: Optional[str] = None) -> pd.DataFrame:
    """Load stocks CSV safely.

    Behavior:
    - Resolves the CSV path to `stocks.csv` in the same folder as this file by default.
    - Does not enforce any schema, but will parse a `Date` column if present.
    """
    csv_path = _project_file("stocks.csv", path)
    logger.info("Loading stocks CSV from: %s", csv_path)
    df = pd.read_csv(csv_path)
    df = _try_parse_dates(df, ("Date",))
    return df


def load_users_csv(path: Optional[str] = None) -> pd.DataFrame:
    """Load users CSV safely.

    Behavior:
    - Resolves the CSV path to `users.csv` in the same folder as this file by default.
    - Does not enforce a schema.
    """
    csv_path = _project_file("users.csv", path)
    logger.info("Loading users CSV from: %s", csv_path)
    return pd.read_csv(csv_path)


def detect_order_transaction_id_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Detect likely OrderID and TransactionID columns in a DataFrame.

    Heuristics (case-insensitive substring matching):
    - Order ID candidates: ["orderid", "order_id", "order id", "order", "order number", "orderno", "order_no"]
    - Transaction ID candidates: ["transactionid", "transaction_id", "transaction id", "transaction", "txn", "txn id", "txn_id"]

    Args:
        df: DataFrame to inspect.

    Returns:
        A dict with keys `order_id_column` and `transaction_id_column` mapping to detected column names or None.
    """
    columns_lower = {c.lower(): c for c in df.columns}

    order_candidates = (
        "orderid",
        "order_id",
        "order id",
        "order",
        "order number",
        "orderno",
        "order_no",
    )
    txn_candidates = (
        "transactionid",
        "transaction_id",
        "transaction id",
        "transaction",
        "txn",
        "txn id",
        "txn_id",
    )

    def _match(candidates: Tuple[str, ...]) -> Optional[str]:
        for needle in candidates:
            for lower_name, original_name in columns_lower.items():
                if needle in lower_name:
                    return original_name
        return None

    order_col = _match(order_candidates)
    txn_col = _match(txn_candidates)

    return {
        "order_id_column": order_col,
        "transaction_id_column": txn_col,
    }


__all__ = [
    "load_sales_csv",
    "load_stocks_csv",
    "load_users_csv",
    "detect_order_transaction_id_columns",
]


