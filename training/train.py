
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    recall_score,
    accuracy_score,
    roc_auc_score
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        default="data/raw/breast_cancer.csv"
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts"
    )

    parser.add_argument(
        "--model-version",
        default="1.0.0"
    )

    args = parser.parse_args()

    # Load parameters
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    seed = params["random_seed"]
    test_size = params["test_size"]
    folds = params["cv_folds"]

    # Load dataset
    df = pd.read_csv(args.data_path)

    feature_names = [
        column for column in df.columns
        if column != "target"
    ]

    X = df[feature_names]

    # Original target: 0 = malignant, 1 = benign
    # Convert to: 1 = malignant, 0 = benign
    y = (df["target"] == 0).astype(int)

    print("Dataset loaded:", df.shape)
    print("Malignant samples:", int(y.sum()))
    print("Benign samples:", int((y == 0).sum()))

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    # Define two different model families
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=params["models"]["logistic_regression"]["C"],
                max_iter=params["models"]["logistic_regression"]["max_iter"],
                random_state=seed
            ))
        ]),

        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=params["models"]["random_forest"]["n_estimators"],
                max_depth=params["models"]["random_forest"]["max_depth"],
                min_samples_split=params["models"]["random_forest"]["min_samples_split"],
                random_state=seed,
                n_jobs=-1
            ))
        ])
    }

    # Five-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed
    )

    scoring = {
        "malignant_recall": "recall",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc"
    }

    results = {}

    best_model_name = None
    best_recall = -1

    print("\nStarting cross-validation...\n")

    for name, pipeline in models.items():

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        cv_results = {
            metric: float(scores[f"test_{metric}"].mean())
            for metric in scoring
        }

        # Fit model on complete training split
        pipeline.fit(X_train, y_train)

        # Evaluate on held-out test split
        predictions = pipeline.predict(X_test)

        malignant_probabilities = pipeline.predict_proba(
            X_test
        )[:, list(pipeline.classes_).index(1)]

        test_results = {
            "malignant_recall": float(
                recall_score(y_test, predictions)
            ),
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "roc_auc": float(
                roc_auc_score(y_test, malignant_probabilities)
            )
        }

        results[name] = {
            "cv": cv_results,
            "test": test_results
        }

        print("Model:", name)
        print("Mean CV metrics:", cv_results)
        print("Test metrics:", test_results)
        print()

        # Select based on mean CV malignant recall
        if cv_results["malignant_recall"] > best_recall:
            best_recall = cv_results["malignant_recall"]
            best_model_name = name
            best_model = pipeline

    # Save selected model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    joblib.dump(best_model, model_path)

    selected_test = results[best_model_name]["test"]

    model_card = {
        "model_version": args.model_version,
        "model_family": best_model_name,
        "feature_names": feature_names,
        "metrics": results[best_model_name],
        "all_models": results,
        "random_seed": seed,
        "sklearn_version": __import__("sklearn").__version__,
        "decision_threshold": params["decision_threshold"]
    }

    with open(output_dir / "model_card.json", "w") as file:
        json.dump(model_card, file, indent=4)

    print("Selected model:", best_model_name)
    print("Mean CV malignant recall:", best_recall)
    print("Test malignant recall:", selected_test["malignant_recall"])
    print("Model saved:", model_path)
    print("Model card saved:", output_dir / "model_card.json")

    # Fail if held-out malignant recall is below requirement
    if selected_test["malignant_recall"] < params["minimum_malignant_recall"]:
        print("ERROR: Test malignant recall is below 0.95")
        raise SystemExit(1)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

