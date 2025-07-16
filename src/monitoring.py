from datetime import datetime, timedelta
from argparse import ArgumentParser

import pandas as pd

import src.config as config
#from src.logger import get_logger
from src.config import FEATURE_GROUP_PREDICTIONS_METADATA, FEATURE_GROUP_METADATA
from src.feature_store_api import get_or_create_feature_group, get_feature_store

#logger = get_logger()


def load_predictions_and_actual_values_from_store(
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    """Fetches model predictions and actuals values from
    `from_date` to `to_date` from the Feature Store and returns a dataframe

    Args:
        from_date (datetime): min datetime for which we want predictions and
        actual values

        to_date (datetime): max datetime for which we want predictions and
        actual values

    Returns:
        pd.DataFrame: 4 columns
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
            - `rides`
    """
    # 2 feature groups we need to merge
    predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
    actuals_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    # query to join the 2 features groups by `pickup_hour` and `pickup_location_id`
    from_ts = int(from_date.timestamp() * 1000)
    to_ts = int(to_date.timestamp() * 1000)
    
    # Try to get predictions data first to check if it exists
    try:
        predictions_data = predictions_fg.read(
            start_time=from_date - timedelta(days=7),
            end_time=to_date + timedelta(days=7)
        )
        print(f"Predictions data shape: {predictions_data.shape}")
        print(f"Predictions columns: {predictions_data.columns.tolist()}")
        has_predictions = len(predictions_data) > 0
    except Exception as e:
        print(f"Error reading predictions data: {e}")
        has_predictions = False
    
    if has_predictions:
        # If we have predictions, do the join
        # Note: actuals_fg uses pickup_hour as timestamp, predictions_fg uses pickup_ts
        query = predictions_fg.select_all() \
            .join(actuals_fg.select(['pickup_location_id', 'pickup_hour', 'rides']),
                  on=['pickup_hour', 'pickup_location_id'], prefix=None) \
            .filter(predictions_fg.pickup_ts >= from_ts) \
            .filter(predictions_fg.pickup_ts <= to_ts)
    else:
        # If no predictions, just return actuals data and add a dummy predictions column
        # Convert pickup_hour to timestamp for filtering
        query = actuals_fg.select_all() \
            .filter(actuals_fg.pickup_hour >= from_date) \
            .filter(actuals_fg.pickup_hour <= to_date)
    
    # breakpoint()

    # create the feature view `config.FEATURE_VIEW_MONITORING` if it does not
    # exist yet
    feature_store = get_feature_store()
    try:
        # create feature view as it does not exist yet
        feature_store.create_feature_view(
            name=config.MONITORING_FV_NAME,
            version=config.MONITORING_FV_VERSION,
            query=query
        )
    except:
        #logger.info('Feature view already existed. Skip creation.')
        print('Feature view already existed. Skip creation.')

    # feature view
    monitoring_fv = feature_store.get_feature_view(
        name=config.MONITORING_FV_NAME,
        version=config.MONITORING_FV_VERSION
    )
    
    # fetch data form the feature view
    # fetch predicted and actual values for the last 30 days
    monitoring_df = monitoring_fv.get_batch_data(
        start_time=from_date - timedelta(days=7),
        end_time=to_date + timedelta(days=7),
    )

    # Debug: print column names to understand the structure
    print(f"Available columns: {monitoring_df.columns.tolist()}")
    print(f"DataFrame shape: {monitoring_df.shape}")
    
    # filter data to the time period we are interested in
    pickup_ts_from = int(from_date.timestamp() * 1000)
    pickup_ts_to = int(to_date.timestamp() * 1000)
    
    # Check if pickup_ts column exists, if not, use alternative filtering
    if 'pickup_ts' in monitoring_df.columns:
        monitoring_df = monitoring_df[monitoring_df.pickup_ts.between(pickup_ts_from, pickup_ts_to)]
    else:
        # Convert pickup_hour to timestamp if pickup_ts is not available
        if 'pickup_hour' in monitoring_df.columns:
            # Convert pickup_hour datetime to timestamp (milliseconds)
            monitoring_df['pickup_ts'] = (monitoring_df['pickup_hour'].astype('int64') // 10**6)
            monitoring_df = monitoring_df[monitoring_df.pickup_ts.between(pickup_ts_from, pickup_ts_to)]
        else:
            print("Warning: Neither pickup_ts nor pickup_hour column found. Using all data.")
            print(f"Available columns: {monitoring_df.columns.tolist()}")
            # Try to filter by date range using datetime comparison
            from_date_utc = from_date.replace(tzinfo=None)
            to_date_utc = to_date.replace(tzinfo=None)
            
            # Look for any datetime column to filter by
            datetime_cols = [col for col in monitoring_df.columns if monitoring_df[col].dtype.kind == 'M']
            if datetime_cols:
                date_col = datetime_cols[0]
                print(f"Using {date_col} for date filtering")
                monitoring_df = monitoring_df[
                    (monitoring_df[date_col] >= from_date_utc) & 
                    (monitoring_df[date_col] <= to_date_utc)
                ]

    # Standardize column names for the frontend
    if 'predictions' in monitoring_df.columns:
        # Rename predictions to predicted_demand for frontend compatibility
        monitoring_df = monitoring_df.rename(columns={'predictions': 'predicted_demand'})
    elif 'predicted_demand' not in monitoring_df.columns:
        # Add dummy predictions column filled with zeros when no predictions exist
        monitoring_df['predicted_demand'] = 0
        print("Added dummy 'predicted_demand' column as no predictions were found")

    return monitoring_df

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--from_date',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--to_date',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()


    monitoring_df = load_predictions_and_actual_values_from_store(args.from_date, args.to_date)