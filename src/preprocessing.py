"""Data preprocessing: cleaning, imputation, outlier handling."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    """Cleans raw Titanic data: imputes missing values, handles outliers."""

    def __init__(self):
        self.age_medians_ = {}
        self.embarked_mode_ = None
        self.fare_median_ = None
        self.age_global_median_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()

        # Calculate medians for Age grouped by Pclass and Sex
        if 'Pclass' in X_df.columns and 'Sex' in X_df.columns and 'Age' in X_df.columns:
            self.age_medians_ = X_df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()

        # Global age median fallback
        if 'Age' in X_df.columns:
            self.age_global_median_ = X_df['Age'].median()

        # Embarked mode
        if 'Embarked' in X_df.columns:
            mode_vals = X_df['Embarked'].mode()
            self.embarked_mode_ = mode_vals[0] if len(mode_vals) > 0 else 'S'

        # Fare median
        if 'Fare' in X_df.columns:
            self.fare_median_ = X_df['Fare'].median()

        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X).copy()

        # Handle Missing Age (grouped median imputation)
        if 'Age' in X_df.columns:
            for key, median_val in self.age_medians_.items():
                pclass, sex = key
                mask = (X_df['Pclass'] == pclass) & (X_df['Sex'] == sex) & (X_df['Age'].isnull())
                X_df.loc[mask, 'Age'] = median_val
            # Fill remaining NaN with global median
            if self.age_global_median_ is not None:
                X_df['Age'] = X_df['Age'].fillna(self.age_global_median_)
            else:
                X_df['Age'] = X_df['Age'].fillna(X_df['Age'].median())

        # Handle Missing Fare
        if 'Fare' in X_df.columns:
            if self.fare_median_ is not None:
                X_df['Fare'] = X_df['Fare'].fillna(self.fare_median_)
            else:
                X_df['Fare'] = X_df['Fare'].fillna(0)

        # Handle Missing Embarked
        if 'Embarked' in X_df.columns and self.embarked_mode_ is not None:
            X_df['Embarked'] = X_df['Embarked'].fillna(self.embarked_mode_)

        # Handle Missing Cabin by filling with 'U' (Unknown)
        if 'Cabin' in X_df.columns:
            X_df['Cabin'] = X_df['Cabin'].fillna('U')

        # Outlier handling — Fare (IQR clipping)
        if 'Fare' in X_df.columns and len(X_df) > 4:
            Q1 = X_df['Fare'].quantile(0.25)
            Q3 = X_df['Fare'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            X_df['Fare'] = np.clip(X_df['Fare'].astype(float), a_min=0, a_max=upper_bound)

        # Outlier handling — Age (Z-score clipping)
        if 'Age' in X_df.columns and len(X_df) > 4:
            age_col = X_df['Age'].astype(float)
            mean = age_col.mean()
            std = age_col.std()
            if std > 0:
                X_df['Age'] = np.clip(age_col, mean - 3 * std, mean + 3 * std)

        return X_df
