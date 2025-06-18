from datetime import datetime
from typing import Tuple

import pandas as pd


def get_default_cutoff_date() -> pd.Timestamp:
    """Returns a default cutoff date for splitting the data into train and test sets.
    Uses January 1st, 2023 as the cutoff date, which splits the data into:
    - Training: 2022 data
    - Test: 2023 data
    """
    return pd.Timestamp('2023-01-01')


def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime = None,
    target_column_name: str = 'rides',
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the data into training and test sets based on a cutoff date.
    
    Args:
        df: DataFrame containing the data
        cutoff_date: Date to split the data into train and test sets. If None, uses January 1st, 2023
        target_column_name: Name of the target column (default: 'rides')
    
    Returns:
        Tuple containing X_train, y_train, X_test, y_test
    """
    if cutoff_date is None:
        cutoff_date = get_default_cutoff_date()
    
    # Convert pickup_hour to datetime if it's not already
    df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])
    
    train_data = df[df.pickup_hour < cutoff_date].reset_index(drop=True)
    test_data = df[df.pickup_hour >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test