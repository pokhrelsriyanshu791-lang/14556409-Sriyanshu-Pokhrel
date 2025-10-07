from __future__ import annotations

import pandas as pd

import utils_pop


def top_sellers() -> pd.DataFrame:
    df = utils_pop.load_sales_csv()
    if df.empty or "Item" not in df.columns:
        return pd.DataFrame(columns=["Item", "Quantity"])  # empty result
    sellers = (
        df.groupby("Item")["Quantity"].sum().reset_index().rename(columns={"Quantity": "Quantity"})
    )
    sellers = sellers.sort_values("Quantity", ascending=False).reset_index(drop=True)
    return sellers


def slow_movers(threshold: int = 5) -> pd.DataFrame:
    df = utils_pop.load_sales_csv()
    if df.empty or "Item" not in df.columns or "Date" not in df.columns:
        return pd.DataFrame(columns=["Item", "Quantity_30d"])  # empty result

    # Consider last 30 days relative to the most recent date in the dataset
    max_date = pd.to_datetime(df["Date"], errors="coerce").max()
    if pd.isna(max_date):
        return pd.DataFrame(columns=["Item", "Quantity_30d"])  # no valid dates

    window_start = max_date.normalize() - pd.Timedelta(days=29)
    mask = (df["Date"] >= window_start) & (df["Date"] <= max_date.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    last_30 = df.loc[mask]

    qty_30 = last_30.groupby("Item")["Quantity"].sum().reset_index().rename(columns={"Quantity": "Quantity_30d"})
    # Include items that had zero sales in 30 days (appear as missing). We'll treat missing as 0 by merging with all items.
    all_items = df[["Item"]].drop_duplicates()
    merged = all_items.merge(qty_30, on="Item", how="left").fillna({"Quantity_30d": 0})
    slow = merged[merged["Quantity_30d"] < threshold].sort_values(["Quantity_30d", "Item"]).reset_index(drop=True)
    return slow


def daily_totals() -> pd.DataFrame:
    df = utils_pop.load_sales_csv()
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame(columns=["Date", "Total_Sales"])  # empty result
    # Monetary daily totals; fall back gracefully if Total Price missing
    if "Total Price" in df.columns:
        amount = pd.to_numeric(df["Total Price"], errors="coerce").fillna(0)
    else:
        # Fallback to quantity sum as a proxy
        amount = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0)
    daily = (
        df.assign(_amount=amount, _date=df["Date"].dt.normalize())
          .groupby("_date")["_amount"].sum().reset_index()
          .rename(columns={"_date": "Date", "_amount": "Total_Sales"})
          .sort_values("Date")
          .reset_index(drop=True)
    )
    return daily


def main() -> None:
    sellers = top_sellers()
    print("Top 5 sellers:")
    if sellers.empty:
        print("No sales data available.")
    else:
        print(sellers.head(5).to_string(index=False))

    print("\nSlow movers (last 30 days, threshold = 5):")
    slow = slow_movers(threshold=5)
    if slow.empty:
        print("None or insufficient recent sales data.")
    else:
        print(slow.to_string(index=False))

    print("\nDaily totals (last 10 days):")
    daily = daily_totals()
    if daily.empty:
        print("No daily totals available.")
    else:
        print(daily.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()


