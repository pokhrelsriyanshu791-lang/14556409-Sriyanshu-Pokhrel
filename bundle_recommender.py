from __future__ import annotations

import itertools
import sys
from typing import List, Tuple

import pandas as pd

import utils_pop


def _extract_baskets(df: pd.DataFrame) -> List[List[str]]:
    """Return a list of baskets, each basket is a list of item names.

    Baskets are grouped by a detected OrderID-like column. If none exists,
    fall back to grouping by a date column if present; otherwise by all rows
    as a single basket.
    """
    # Prefer an explicit Order/Transaction ID if present
    id_info = utils_pop.detect_order_transaction_id_columns(df)
    order_col = id_info.get("order_id_column")

    if order_col and order_col in df.columns:
        group_key = order_col
    else:
        # Fallback to a date-like column that utils_pop.load_sales_csv may have parsed
        date_candidates = ("Date", "OrderDate", "TransactionDate", "SaleDate")
        group_key = next((c for c in date_candidates if c in df.columns), None)

    if group_key is None:
        # No reasonable grouping column; treat entire df as a single basket
        return [df["Item"].astype(str).tolist()] if "Item" in df.columns else []

    baskets: List[List[str]] = []
    for _, group in df.groupby(group_key):
        if "Item" not in group.columns:
            continue
        items = group["Item"].astype(str).tolist()
        # Deduplicate items within the same basket to avoid overcounting
        unique_items = list(dict.fromkeys(items))
        if unique_items:
            baskets.append(unique_items)
    return baskets


def _count_cooccurrences(baskets: List[List[str]]) -> Tuple[dict[Tuple[str, str], int], dict[str, int]]:
    """Build co-occurrence counts for unordered product pairs across baskets.
    
    Returns:
        Tuple of (pair_counts, item_counts) where:
        - pair_counts: dict mapping (item_a, item_b) to co-occurrence count
        - item_counts: dict mapping item to individual occurrence count
    """
    pair_counts: dict[Tuple[str, str], int] = {}
    item_counts: dict[str, int] = {}
    
    for items in baskets:
        if len(items) < 2:
            continue
        # Count individual items
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1
        
        # Generate all unordered pairs without repetition
        for a, b in itertools.combinations(sorted(items), 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1
    
    return pair_counts, item_counts


def generate_bundles(min_support: int = 2) -> pd.DataFrame:
    """Generate bundle suggestions based on association rules.

    - Loads sales using utils_pop.load_sales_csv()
    - Groups by detected OrderID; if none, groups by Date
    - Calculates support, confidence, and lift for item pairs
    - Filters by min_support
    - Saves to bundle_suggestions.csv (Product_A, Product_B, Count, Confidence, Lift)

    Returns the resulting DataFrame.
    """
    sales_df = utils_pop.load_sales_csv()

    if sales_df.empty:
        result_df = pd.DataFrame(columns=["Product_A", "Product_B", "Count", "Confidence", "Lift"])
        result_df.to_csv("bundle_suggestions.csv", index=False)
        return result_df

    baskets = _extract_baskets(sales_df)
    pair_counts, item_counts = _count_cooccurrences(baskets)
    
    # Calculate association rules
    results = []
    total_baskets = len(baskets)
    
    for (item_a, item_b), pair_count in pair_counts.items():
        if pair_count < min_support:
            continue
            
        # Support = how often the pair appears together
        support = pair_count / total_baskets
        
        # Confidence = P(B|A) = P(A and B) / P(A)
        # For unordered pairs, we calculate both directions
        item_a_count = item_counts.get(item_a, 0)
        item_b_count = item_counts.get(item_b, 0)
        
        if item_a_count > 0:
            confidence_a_to_b = pair_count / item_a_count
            # Lift = P(A and B) / (P(A) * P(B))
            lift_a_to_b = (pair_count / total_baskets) / ((item_a_count / total_baskets) * (item_b_count / total_baskets)) if item_b_count > 0 else 0
            
            results.append({
                'Product_A': item_a,
                'Product_B': item_b,
                'Count': pair_count,
                'Confidence': confidence_a_to_b,
                'Lift': lift_a_to_b
            })
        
        if item_b_count > 0:
            confidence_b_to_a = pair_count / item_b_count
            # Lift = P(A and B) / (P(A) * P(B))
            lift_b_to_a = (pair_count / total_baskets) / ((item_a_count / total_baskets) * (item_b_count / total_baskets)) if item_a_count > 0 else 0
            
            results.append({
                'Product_A': item_b,
                'Product_B': item_a,
                'Count': pair_count,
                'Confidence': confidence_b_to_a,
                'Lift': lift_b_to_a
            })

    # Convert to DataFrame and sort by confidence descending
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('Confidence', ascending=False)

    # Persist results
    result_df.to_csv("bundle_suggestions.csv", index=False)
    return result_df


def main() -> None:
    # Allow passing min_support from command line
    if len(sys.argv) > 1:
        try:
            min_support = int(sys.argv[1])
        except ValueError:
            print("Invalid min_support value, using default = 2")
            min_support = 2
    else:
        min_support = 2

    df = generate_bundles(min_support=min_support)
    # Print top 5 preview
    if df.empty:
        print("No bundle suggestions found.")
        return
    preview = df.head(5)
    print("Top 5 bundle suggestions (min_support =", min_support, "):")
    for _, row in preview.iterrows():
        confidence_pct = row['Confidence'] * 100
        print(f"{row['Product_A']} -> {row['Product_B']} (Count: {row['Count']}, Confidence: {confidence_pct:.1f}%, Lift: {row['Lift']:.2f})")


if __name__ == "__main__":
    main()


