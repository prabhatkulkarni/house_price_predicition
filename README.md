# House Price Prediction with Linear Regression

A small regression project exploring how a few property attributes relate to sale price in the King County, Washington housing dataset.

The script compares three single-feature linear regression models against a model trained with all three features.

## Problem

Predict a property's sale price from:
- `bedrooms`
- `bathrooms`
- `sqft_living`

The target is `price`. The repository includes the dataset as `data.csv`.

## What the script does

1. Loads the CSV with pandas.
2. Splits the data into training and test sets.
3. Trains one model for each individual feature.
4. Trains a multi-feature model using bedrooms, bathrooms, and living area.
5. Compares models using the coefficient of determination (`R²`).
6. Plots actual versus predicted values and single-feature regression lines.

The same train/test split is used for the four models so their R² scores are comparable. A higher R² indicates that the model explains more of the variation in the held-out target values; it does not guarantee accurate predictions for every property.

## Requirements

- Python 3
- pandas
- NumPy
- scikit-learn
- Matplotlib

Install dependencies:

```bash
python -m pip install pandas numpy scikit-learn matplotlib
```

## Run

Clone the repository and run these commands from its root directory, where `data.csv` is located:

```bash
python main.py
```

The script prints the R² scores and opens plots for the single-feature comparisons and the multi-feature model.

## Repository structure

```text
house_price_predicition/
├── data.csv
├── main.py
└── README.md
```

## Current limitations

- Only three numeric features are used; location, condition, year built, and other factors are not included.
- Linear regression cannot capture every nonlinear relationship in housing data.
- R² alone is not a complete evaluation. Future work could report MAE and RMSE, compare against a baseline, inspect residuals, and validate on additional splits.
- The analysis is educational and is not intended for real-estate valuation.

## Next steps

- [x] Compare single-feature and multi-feature linear regression
- [x] Use a consistent hold-out split for model comparisons
- [ ] Add MAE and RMSE
- [ ] Add residual analysis
- [ ] Compare with a regularized model and a tree-based baseline
