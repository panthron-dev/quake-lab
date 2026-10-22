"""Task 5: split and preprocess."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, PolynomialFeatures,
                                   RobustScaler, StandardScaler)

from . import config

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test.

    Stratified on config.TARGET, seeded with config.SEED.
    Neither the target nor 'mag' may remain in X.
    """
    # 'mag' IS the answer (big_quake = mag >= 4.5), so it must leave X.
    X = df.drop(columns=[c for c in (config.TARGET, "mag") if c in df.columns])
    y = df[config.TARGET]
    # stratify=y keeps the rare positive class at the same rate in both halves.
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=config.SEED)


def build_preprocessor(scaler: str = "robust", degree: int = 1) -> ColumnTransformer:
    """ColumnTransformer over config.NUMERIC and config.NOMINAL.

    numeric: median imputation, then a scaler chosen by name
             ('standard', 'minmax', 'robust'); when degree > 1, polynomial
             features are added between imputation and scaling (Part 2).
    nominal: most-frequent imputation, then one-hot (handle_unknown='ignore')
    """
    if scaler not in SCALERS:
        raise ValueError(f"scaler must be one of {list(SCALERS)}, got {scaler!r}")
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if degree > 1:
        num_steps.append(("poly", PolynomialFeatures(degree, include_bias=False)))
    num_steps.append(("scale", SCALERS[scaler]()))

    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a region never seen in training becomes all
        # zeros instead of crashing at prediction time.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), config.NUMERIC),
        ("nom", nominal, config.NOMINAL),
    ], remainder="drop")  # id, url, place... are not features
