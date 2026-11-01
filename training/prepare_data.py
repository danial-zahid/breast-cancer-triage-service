
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def main():
    # Load the built-in breast cancer dataset
    dataset = load_breast_cancer()

    # Convert feature names into API-friendly column names
    feature_names = [
        name.replace(" ", "_")
        for name in dataset.feature_names
    ]

    # Create DataFrame containing all 30 features
    df = pd.DataFrame(
        dataset.data,
        columns=feature_names
    )

    # Add target column
    df["target"] = dataset.target

    # Validate dataset
    assert len(feature_names) == 30, "Feature count must be 30"
    assert not df.isnull().values.any(), "Dataset contains null values"

    # Create output directory
    output_path = Path("data/raw/breast_cancer.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export CSV
    df.to_csv(output_path, index=False)

    # Display dataset information
    print("Dataset exported successfully!")
    print(f"File: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Feature columns: {len(feature_names)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Null values: {df.isnull().sum().sum()}")
    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index())


if __name__ == "__main__":
    main()
