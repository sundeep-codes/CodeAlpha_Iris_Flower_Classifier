import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import FEATURE_COLUMNS, TARGET_COLUMN

def inspect_dataset(df):
    """
    Prints shape, dtypes, missing values, duplicates, summary statistics.
    Returns a dict with all inspection results.
    """
    results = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing_values': check_missing_values(df),
        'duplicates': check_duplicates(df),
        'summary_statistics': get_summary_statistics(df)
    }
    
    print(f"Shape: {results['shape']}")
    print(f"Missing Values:\n{pd.Series(results['missing_values'])}")
    print(f"Duplicates: {results['duplicates']}")
    print("Summary Statistics:")
    print(results['summary_statistics'])
    
    return results

def check_missing_values(df):
    """Returns missing value counts as a dictionary."""
    return df.isnull().sum().to_dict()

def check_duplicates(df):
    """Returns number of duplicates."""
    return int(df.duplicated().sum())

def get_summary_statistics(df):
    """Returns df.describe()."""
    return df.describe()

def prepare_data(df, test_size=0.3, random_state=12):
    """
    Splits into X_train, X_test, y_train, y_test using train_test_split.
    Uses StandardScaler to scale features.
    Returns X_train, X_test, y_train, y_test, scaler
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS, index=X_test.index)
    
    return X_train, X_test, y_train, y_test, scaler
