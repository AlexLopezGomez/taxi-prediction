from datetime import datetime, timedelta

import hopsworks
from hsfs.feature_store import FeatureStore
from hsfs.feature_view import FeatureView
import pandas as pd
import numpy as np

import src.config as config
# from src.feature_store_api import get_feature_store, get_or_create_feature_view
# from src.config import FEATURE_VIEW_METADATA

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )


def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store(name=config.HOPSWORKS_PROJECT_NAME)


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """"""
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)
    
    return results


def load_batch_of_features_from_store(
    current_date: pd.Timestamp,    
) -> pd.DataFrame:
    """Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 4 columns:
            - `pickup_hour`
            - `rides_count`
            - `pickup_location_id`
            - `pickpu_ts`
    """
    n_features = config.N_FEATURES

    #feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)
   # project = get_hopsworks_project()
    #feature_view = FeatureView(project.get_feature_store(), name=config.FEATURE_VIEW_NAME)
    feature_store = get_feature_store()

    feature_view = feature_store.get_feature_view(name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION)

    # Obtener la última fecha disponible en el feature store
    ts_data_full = feature_view.get_batch_data(query_service=True)
    max_date = ts_data_full['pickup_hour'].max()
    if current_date > max_date:
        print(f"[INFO] La fecha pedida ({current_date}) está fuera del rango disponible. Usando la última fecha disponible: {max_date}.")
        current_date = max_date

    # fetch data from the feature store
    fetch_data_from = current_date - timedelta(days=28)
    fetch_data_to = current_date - timedelta(hours=1)

    # add plus minus margin to make sure we do not drop any observation
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(days=1),
        end_time=fetch_data_to + timedelta(days=1),
        query_service=True
    )
    
    # filter data to the time period we are interested in
    pickup_hour_from = fetch_data_from
    pickup_hour_to = fetch_data_to
    ts_data = ts_data[ts_data['pickup_hour'].between(pickup_hour_from, pickup_hour_to)]
    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()
    assert len(ts_data) == config.N_FEATURES * len(location_ids), \
        "Time-series data is not complete. Make sure your feature pipeline is up and runnning."

    # transpose time-series data as a feature vector, for each `pickup_location_id`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides_count'].values

    # numpy arrays to Pandas dataframes
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features


def load_model_from_registry():
    
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )  
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / 'model.pkl')
       
    return model

