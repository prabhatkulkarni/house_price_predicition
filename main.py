"""Compare single-feature and multi-feature linear regression models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


DATA_PATH = Path(__file__).resolve().parent / "data.csv"
FEATURES = ["bedrooms", "bathrooms", "sqft_living"]
TARGET = "price"


def main() -> None:
    """Load data, train four models, report R², and plot predictions."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run this script from the repository "
            "or keep data.csv in the same directory as main.py."
        )

    house_df = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in house_df.columns]
    if missing_columns:
        raise ValueError(f"data.csv is missing required columns: {missing_columns}")

    # Use one shared split so all models are evaluated on the same held-out rows.
    X = house_df[FEATURES]
    y = house_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    single_models = {}
    single_predictions = {}
    scores = {}

    for feature in FEATURES:
        model = LinearRegression()
        model.fit(X_train[[feature]], y_train)
        prediction = model.predict(X_test[[feature]])

        single_models[feature] = model
        single_predictions[feature] = prediction
        scores[f"{feature} only"] = r2_score(y_test, prediction)

    multi_model = LinearRegression()
    multi_model.fit(X_train, y_train)
    multi_prediction = multi_model.predict(X_test)
    scores["All three features"] = r2_score(y_test, multi_prediction)

    print("Held-out R² scores")
    print("------------------")
    for name, score in scores.items():
        print(f"{name:24s}: {score:.4f}")

    # Show each one-feature model against the same actual test targets.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, feature in zip(axes, FEATURES):
        x_test = X_test[feature].to_numpy()
        order = np.argsort(x_test)

        ax.scatter(x_test, y_test, alpha=0.55, label="Actual")
        ax.plot(
            x_test[order],
            single_predictions[feature][order],
            label="Linear regression",
        )
        ax.set_title(f"{feature} only (R²={scores[f'{feature} only']:.3f})")
        ax.set_xlabel(feature)
        ax.set_ylabel("Sale price (USD)")
        ax.legend()

    fig.suptitle("Single-feature linear regression")
    fig.tight_layout()
    plt.show()

    # Compare multi-feature predictions against actual held-out prices.
    lower = min(float(y_test.min()), float(np.min(multi_prediction)))
    upper = max(float(y_test.max()), float(np.max(multi_prediction)))

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, multi_prediction, alpha=0.55)
    plt.plot([lower, upper], [lower, upper], linestyle="--", label="Ideal prediction")
    plt.xlabel("Actual sale price (USD)")
    plt.ylabel("Predicted sale price (USD)")
    plt.title(f"Multi-feature model (R²={scores['All three features']:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
