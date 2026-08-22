import pandas as pd


def main():
    variable = pd.Series([
        "10", "20", "30", "40", "50",
        "60", "70", "80", "90", "100",
        "110", "120", "130", "140",
        None, None,
        "Paris", "Abidjan", "France", "Unknown"
    ])

    # Identify values that are not missing in the original Series
    non_missing = variable.notna()

    # Convert to numeric, non-convertible values become NaN
    numeric = pd.to_numeric(variable, errors="coerce")

    # Conversion rate among originally non-missing values
    conversion_rate = numeric[non_missing].notna().mean()

    print("Original Series:")
    print(variable)

    print("\nNumeric Series:")
    print(numeric)

    print(f"\nNumber of observations: {len(variable)}")
    print(f"Number of non-missing values: {non_missing.sum()}")
    print(f"Number of successfully converted values: {numeric[non_missing].notna().sum()}")
    print(f"Conversion rate: {conversion_rate:.2%}")


if __name__ == "__main__":
    main()