import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        
        # Convert Pclass to string for OneHotEncoding
        if 'Pclass' in X_df.columns:
            X_df['Pclass'] = X_df['Pclass'].astype(str)
        
        if 'SibSp' in X_df.columns and 'Parch' in X_df.columns:
            X_df['FamilySize'] = X_df['SibSp'].astype(float) + X_df['Parch'].astype(float) + 1
            X_df['IsAlone'] = (X_df['FamilySize'] == 1).astype(int)
            
        if 'Name' in X_df.columns:
            X_df['Title'] = X_df['Name'].apply(self.extract_title)
            
        if 'Ticket' in X_df.columns:
            ticket_counts = X_df['Ticket'].value_counts()
            X_df['TicketGroupSize'] = X_df['Ticket'].map(ticket_counts).fillna(1).astype(float)
            
        if 'Cabin' in X_df.columns:
            X_df['CabinDeck'] = X_df['Cabin'].astype(str).str[0]
            
        if 'Fare' in X_df.columns and 'FamilySize' in X_df.columns:
            X_df['FarePerPerson'] = X_df['Fare'].astype(float) / X_df['FamilySize'].replace(0, 1)
            
        if 'Age' in X_df.columns:
            age_float = X_df['Age'].astype(float)
            X_df['AgeGroup'] = pd.cut(age_float, bins=[-1, 12, 20, 40, 60, 200], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']).astype(str)
            X_df['IsChild'] = (age_float < 12).astype(int)
            
        if 'Fare' in X_df.columns:
            try:
                X_df['FareCategory'] = pd.qcut(X_df['Fare'].astype(float).rank(method='first'), 4, labels=['Low', 'Mid', 'High', 'VeryHigh']).astype(str)
            except ValueError:
                X_df['FareCategory'] = 'Mid'
            
        # Drop columns not needed anymore
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        X_df = X_df.drop(columns=[c for c in cols_to_drop if c in X_df.columns])
        
        return X_df
        
    def extract_title(self, name):
        if not isinstance(name, str): return 'Unknown'
        title_search = re.search(r' ([A-Za-z]+)\.', name)
        if title_search:
            title = title_search.group(1)
            if title in ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
                return 'Rare'
            if title == 'Mlle':
                return 'Miss'
            if title == 'Ms':
                return 'Miss'
            if title == 'Mme':
                return 'Mrs'
            return title
        return 'Unknown'

class SafeColumnTransformer(BaseEstimator, TransformerMixin):
    """Column transformer that handles missing columns gracefully. Module-level for pickling."""

    def __init__(self, num_feats=None, cat_feats=None, bin_feats=None):
        self.num_feats = num_feats or []
        self.cat_feats = cat_feats or []
        self.bin_feats = bin_feats or []
        self.ct_ = None
        self.feature_names_out_ = None
        self.num_ = None
        self.cat_ = None
        self.bin_ = None

    def fit(self, X, y=None):
        X_cols = X.columns if hasattr(X, 'columns') else list(range(X.shape[1]))
        self.num_ = [c for c in self.num_feats if c in X_cols]
        self.cat_ = [c for c in self.cat_feats if c in X_cols]
        self.bin_ = [c for c in self.bin_feats if c in X_cols]

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.ct_ = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_),
                ('cat', categorical_transformer, self.cat_),
                ('bin', 'passthrough', self.bin_)
            ]
        )
        self.ct_.fit(X, y)

        cat_feature_names = list(self.ct_.named_transformers_['cat']
                                 .named_steps['onehot']
                                 .get_feature_names_out(self.cat_))
        self.feature_names_out_ = self.num_ + cat_feature_names + self.bin_
        return self

    def transform(self, X, y=None):
        return self.ct_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_) if self.feature_names_out_ else np.array([])


def get_preprocessor():
    from src.preprocessing import DataCleaner

    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'TicketGroupSize', 'FarePerPerson']
    categorical_features = ['Sex', 'Embarked', 'Title', 'CabinDeck', 'AgeGroup', 'FareCategory', 'Pclass']
    binary_features = ['IsAlone', 'IsChild']

    preprocessor = Pipeline([
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()),
        ('col_trans', SafeColumnTransformer(numeric_features, categorical_features, binary_features))
    ])

    return preprocessor

if __name__ == "__main__":
    from src.data_loader import load_data
    df = load_data()
    X = df.drop('Survived', axis=1, errors='ignore')
    preprocessor = get_preprocessor()
    X_proc = preprocessor.fit_transform(X)
    print("Processed shape:", X_proc.shape)
