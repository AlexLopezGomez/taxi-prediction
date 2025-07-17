from datetime import datetime
from typing import Tuple

import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the data into training and test sets.
    """
    # Check if dataframe is empty
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot perform train/test split. Please check if the feature view contains data.")
    
    # Ensure pickup_hour is datetime format
    if df['pickup_hour'].dtype == 'object':
        # If pickup_hour is string, convert to datetime
        df = df.copy()
        df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])
    
    # Ensure cutoff_date is timezone-aware if pickup_hour has timezone
    if len(df) > 0 and hasattr(df['pickup_hour'].iloc[0], 'tz') and df['pickup_hour'].iloc[0].tz is not None:
        # pickup_hour has timezone, ensure cutoff_date has the same timezone
        if cutoff_date.tzinfo is None:
            cutoff_date = pd.Timestamp(cutoff_date, tz=df['pickup_hour'].iloc[0].tz)
    elif hasattr(cutoff_date, 'tz') and cutoff_date.tz is not None:
        # cutoff_date has timezone but pickup_hour doesn't, remove timezone from cutoff_date
        cutoff_date = cutoff_date.tz_localize(None)
    
    train_data = df[df.pickup_hour < cutoff_date].reset_index(drop=True)
    test_data = df[df.pickup_hour >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test