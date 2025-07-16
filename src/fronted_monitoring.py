from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error
import plotly.express as px

from src.monitoring import load_predictions_and_actual_values_from_store


st.set_page_config(layout="wide")

# title
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Monitoring dashboard 🔎')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 3


# @st.cache_data
def _load_predictions_and_actuals_from_store(
    from_date: datetime,
    to_date: datetime
    ) -> pd.DataFrame:
    """Wrapped version of src.monitoring.load_predictions_and_actual_values_from_store, so
    we can add Streamlit caching

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
    return load_predictions_and_actual_values_from_store(from_date, to_date)

with st.spinner(text="Fetching model predictions and actual values from the store"):
    
    monitoring_df = _load_predictions_and_actuals_from_store(
        from_date=current_date - timedelta(days=80),
        to_date=current_date
    )
    st.sidebar.write('✅ Model predictions and actual values arrived')
    progress_bar.progress(1/N_STEPS)
    
    # Debug: Show DataFrame structure
    st.write(f"DataFrame shape: {monitoring_df.shape}")
    st.write(f"Available columns: {monitoring_df.columns.tolist()}")
    
    # Check if we have the required columns
    if 'predicted_demand' not in monitoring_df.columns:
        st.error("⚠️ No predictions found in the feature store. The 'predicted_demand' column is missing.")
        st.write("Available columns:", monitoring_df.columns.tolist())
        st.stop()
    
    if len(monitoring_df) == 0:
        st.error("⚠️ No data found in the specified date range.")
        st.stop()
    
    # Check if predictions are all zeros (dummy predictions)
    if monitoring_df['predicted_demand'].nunique() == 1 and monitoring_df['predicted_demand'].iloc[0] == 0:
        st.warning("⚠️ No real predictions found in the feature store. Showing dummy predictions (all zeros) for visualization purposes.")
        st.write("To get real predictions, run the prediction pipeline to populate the predictions feature group.")
        
    print(monitoring_df.head())


with st.spinner(text="Plotting aggregate MAE hour-by-hour"):
    
    st.header('Mean Absolute Error (MAE) hour-by-hour')

    # Convert pickup_hour to hour of day for grouping if it's a datetime
    if monitoring_df['pickup_hour'].dtype.kind == 'M':  # datetime type
        monitoring_df['hour_of_day'] = monitoring_df['pickup_hour'].dt.hour
        group_column = 'hour_of_day'
    else:
        group_column = 'pickup_hour'

    # MAE per pickup_hour
    # https://stackoverflow.com/a/47914634
    mae_per_hour = (
        monitoring_df
        .groupby(group_column)
        .apply(lambda g: mean_absolute_error(g['rides'], g['predicted_demand']))
        .reset_index()
        .rename(columns={0: 'mae'})
        .sort_values(by=group_column)
    )

    fig = px.bar(
        mae_per_hour,
        x=group_column, y='mae',
        template='plotly_dark',
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(2/N_STEPS)


with st.spinner(text="Plotting MAE hour-by-hour for top locations"):
    
    st.header('Mean Absolute Error (MAE) per location and hour')

    top_locations_by_demand = (
        monitoring_df
        .groupby('pickup_location_id')['rides']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .head(10)['pickup_location_id']
    )

    for location_id in top_locations_by_demand:
        
        mae_per_hour = (
            monitoring_df[monitoring_df.pickup_location_id == location_id]
            .groupby(group_column)
            .apply(lambda g: mean_absolute_error(g['rides'], g['predicted_demand']))
            .reset_index()
            .rename(columns={0: 'mae'})
            .sort_values(by=group_column)
        )

        fig = px.bar(
            mae_per_hour,
            x=group_column, y='mae',
            template='plotly_dark',
        )
        st.subheader(f'{location_id=}')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(3/N_STEPS)