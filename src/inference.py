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
            - `rides`
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
        x[i, :] = ts_data_i['rides'].values

    # numpy arrays to Pandas dataframes
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features


def load_batch_of_features_from_store_v2(
    current_date: pd.Timestamp,    
) -> pd.DataFrame:
    """
    Versión mejorada de load_batch_of_features_from_store que maneja casos donde
    no hay suficientes datos en el feature store.
    
    Esta función es útil para desarrollo/testing cuando el pipeline de datos
    no está completamente actualizado.

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: n_features + 2 columns:
            - `rides_previous_N_hour`
            - `rides_previous_{N-1}_hour`
            - ...
            - `rides_previous_1_hour`
            - `pickup_hour`
            - `pickup_location_id`
    """
    n_features = config.N_FEATURES

    feature_store = get_feature_store()
    feature_view = feature_store.get_feature_view(name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION)

    # Obtener la última fecha disponible en el feature store
    ts_data_full = feature_view.get_batch_data(query_service=True)
    max_date = ts_data_full['pickup_hour'].max()
    min_date = ts_data_full['pickup_hour'].min()
    
    print(f"[INFO] Rango de datos disponibles: {min_date} a {max_date}")
    
    # Ajustar current_date si está fuera del rango disponible
    if current_date > max_date:
        print(f"[INFO] La fecha pedida ({current_date}) está fuera del rango disponible. Usando la última fecha disponible: {max_date}.")
        current_date = max_date

    # Calcular el rango de fechas que necesitamos
    ideal_fetch_from = current_date - timedelta(days=28)
    fetch_data_to = current_date - timedelta(hours=1)
    
    # Verificar si tenemos suficientes datos históricos
    if ideal_fetch_from < min_date:
        print(f"[WARNING] No hay suficientes datos históricos.")
        print(f"[WARNING] Ideal: desde {ideal_fetch_from}, Disponible: desde {min_date}")
        fetch_data_from = min_date
        
        # Calcular cuántos días de datos tenemos realmente
        days_available = (max_date - min_date).days + 1
        print(f"[INFO] Días de datos disponibles: {days_available}")
        
        if days_available < 7:
            print(f"[WARNING] Solo hay {days_available} días de datos. Usando todo el rango disponible.")
            fetch_data_to = max_date
        else:
            # Usar los últimos días disponibles (máximo 14 días si no tenemos 28)
            days_to_use = min(days_available - 1, 14)
            fetch_data_from = max_date - timedelta(days=days_to_use)
            fetch_data_to = max_date
            print(f"[INFO] Usando los últimos {days_to_use} días de datos.")
    else:
        fetch_data_from = ideal_fetch_from

    # Obtener datos del feature store con margen
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(hours=1),
        end_time=fetch_data_to + timedelta(hours=1),
        query_service=True
    )
    
    # Filtrar al período exacto que necesitamos
    ts_data = ts_data[ts_data['pickup_hour'].between(fetch_data_from, fetch_data_to)]
    
    # Ordenar datos por ubicación y tiempo
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

    # Obtener todas las ubicaciones únicas
    location_ids = ts_data['pickup_location_id'].unique()
    
    # Calcular cuántas horas de datos tenemos realmente
    hours_available = len(ts_data['pickup_hour'].unique()) if len(ts_data) > 0 else 0
    actual_features = min(n_features, hours_available)
    
    print(f"[INFO] Horas de datos disponibles: {hours_available}")
    print(f"[INFO] Features requeridas: {n_features}, Features que usaremos: {actual_features}")
    
    if actual_features < n_features:
        print(f"[WARNING] Faltan {n_features - actual_features} horas de datos. Se rellenará con ceros.")
    
    # Verificar integridad de datos de manera más flexible
    expected_total_rows = actual_features * len(location_ids)
    if len(ts_data) < expected_total_rows:
        print(f"[WARNING] Datos incompletos. Esperado: {expected_total_rows}, Encontrado: {len(ts_data)}")
        print(f"[INFO] Continuando con datos disponibles y rellenando faltantes con ceros.")

    # Crear matriz de features con ceros
    x = np.zeros(shape=(len(location_ids), n_features), dtype=np.float32)
    
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        
        # Obtener los valores disponibles
        available_values = ts_data_i['rides'].values
        
        if len(available_values) > 0:
            # Si tenemos menos datos de los necesarios, colocar los datos más recientes al final
            if len(available_values) < n_features:
                # Los datos más recientes van al final del array (más importantes)
                x[i, n_features-len(available_values):] = available_values
            else:
                # Si tenemos suficientes datos, tomar los últimos n_features
                x[i, :] = available_values[-n_features:]

    # Convertir numpy arrays a DataFrame
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    print(f"[SUCCESS] Features generadas para {len(location_ids)} ubicaciones con {actual_features}/{n_features} horas de datos.")

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

