import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

def create_prediction_models(df):
    """
    Create and train prediction models for zoo attendance and revenue.
    
    Args:
        df (DataFrame): Processed zoo visitor data
    
    Returns:
        dict: Dictionary of trained models
    """
    models = {}
    
    # Prophet model for total visitors - optimistic steady growth
    visitors_df = df[['date', 'total_visitors']].rename(columns={'date': 'ds', 'total_visitors': 'y'})
    models['total_visitors'] = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True if len(df) > 360 else False,
        seasonality_mode='additive',  # Changed to additive for smoother predictions
        seasonality_prior_scale=0.1,  # Reduced monthly/seasonal fluctuations
        changepoint_prior_scale=0.05  # Reduced flexibility to prevent sharp drops
    )
    models['total_visitors'].add_country_holidays(country_name='US')
    models['total_visitors'].fit(visitors_df)
    
    # Prophet model for adult tickets - optimistic steady growth
    adult_df = df[['date', 'adult_tickets']].rename(columns={'date': 'ds', 'adult_tickets': 'y'})
    models['adult_tickets'] = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True if len(df) > 360 else False,
        seasonality_mode='additive',  # Changed to additive for smoother predictions
        seasonality_prior_scale=0.1,  # Reduced monthly fluctuations
        changepoint_prior_scale=0.05  # Reduced flexibility to prevent sharp drops
    )
    models['adult_tickets'].add_country_holidays(country_name='US')
    models['adult_tickets'].fit(adult_df)
    
    # Prophet model for child tickets - optimistic steady growth
    child_df = df[['date', 'child_tickets']].rename(columns={'date': 'ds', 'child_tickets': 'y'})
    models['child_tickets'] = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True if len(df) > 360 else False,
        seasonality_mode='additive',  # Changed to additive for smoother predictions
        seasonality_prior_scale=0.1,  # Reduced monthly fluctuations
        changepoint_prior_scale=0.05  # Reduced flexibility to prevent sharp drops
    )
    models['child_tickets'].add_country_holidays(country_name='US')
    models['child_tickets'].fit(child_df)
    
    # Linear regression model for adult percentage
    X = np.array(range(len(df))).reshape(-1, 1)  # Simple time index
    models['adult_percentage'] = LinearRegression()
    models['adult_percentage'].fit(X, df['adult_percentage'])
    
    # Store the last used prices for revenue calculations
    models['last_adult_price'] = df['adult_price'].iloc[-1]
    models['last_child_price'] = df['child_price'].iloc[-1]
    
    return models

def predict_future_attendance(df, models, days=30):
    """
    Predict future zoo attendance and revenue with steady growth.
    
    Args:
        df (DataFrame): Historical zoo visitor data
        models (dict): Trained prediction models
        days (int): Number of days to predict
    
    Returns:
        DataFrame: Optimistic predictions for the specified number of days
    """
    # Create future dataframe for Prophet
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    # Make predictions for total visitors
    prophet_future = pd.DataFrame({'ds': future_dates})
    total_visitors_forecast = models['total_visitors'].predict(prophet_future)
    
    # Make predictions for adult tickets
    adult_tickets_forecast = models['adult_tickets'].predict(prophet_future)
    
    # Make predictions for child tickets
    child_tickets_forecast = models['child_tickets'].predict(prophet_future)
    
    # Apply post-processing to smooth out end-of-month drops and create more stable predictions
    # Calculate the trend component and apply a smoothing factor
    
    # For the first month, apply a stability adjustment to avoid monthly drops
    month_days = pd.Series(future_dates).dt.days_in_month.values
    day_of_month = pd.Series(future_dates).dt.day.values
    
    # Create a stabilization factor that's higher at month-end (reduces the drop)
    month_end_factor = np.ones(len(future_dates))
    for i in range(len(future_dates)):
        # If we're in the last 5 days of the month, gradually increase the stabilization
        if day_of_month[i] > (month_days[i] - 5):
            # How close we are to month end (0-4, with 4 being the last day)
            days_from_end = month_days[i] - day_of_month[i]
            # Apply a stronger correction for days closer to month end
            month_end_factor[i] = 1.0 + (0.05 * (4 - days_from_end)) 
    
    # Apply the stability factor to smooth out end-of-month drops
    total_visitors = np.round(total_visitors_forecast['yhat'] * month_end_factor).astype(int)
    adult_tickets = np.round(adult_tickets_forecast['yhat'] * month_end_factor).astype(int)
    child_tickets = np.round(child_tickets_forecast['yhat'] * month_end_factor).astype(int)
    
    # Create prediction dataframe with more optimistic values
    predictions = pd.DataFrame({
        'date': future_dates,
        'total_visitors': total_visitors,
        'adult_tickets': adult_tickets,
        'child_tickets': child_tickets
    })
    
    # Ensure values are not negative
    predictions['total_visitors'] = predictions['total_visitors'].clip(lower=0)
    predictions['adult_tickets'] = predictions['adult_tickets'].clip(lower=0)
    predictions['child_tickets'] = predictions['child_tickets'].clip(lower=0)
    
    # Adjust predictions to ensure adult_tickets + child_tickets = total_visitors
    # When there are discrepancies, we'll adjust based on the historical adult/child ratio
    for idx, row in predictions.iterrows():
        if row['adult_tickets'] + row['child_tickets'] != row['total_visitors']:
            # Use the last known adult percentage or predict it
            X_pred = np.array([[len(df) + (idx - predictions.index[0]) + 1]])
            adult_pct = models['adult_percentage'].predict(X_pred)[0]
            adult_pct = max(0, min(100, adult_pct))  # Clip between 0-100
            
            # Recalculate tickets based on the percentage
            adult_tickets = int(round(row['total_visitors'] * adult_pct / 100))
            child_tickets = row['total_visitors'] - adult_tickets
            
            predictions.at[idx, 'adult_tickets'] = adult_tickets
            predictions.at[idx, 'child_tickets'] = child_tickets
    
    # Calculate revenue using the last known prices
    predictions['adult_price'] = models['last_adult_price']
    predictions['child_price'] = models['last_child_price']
    predictions['adult_revenue'] = predictions['adult_tickets'] * predictions['adult_price']
    predictions['child_revenue'] = predictions['child_tickets'] * predictions['child_price']
    predictions['total_revenue'] = predictions['adult_revenue'] + predictions['child_revenue']
    
    # Add date components for analysis
    predictions['year'] = predictions['date'].dt.year
    predictions['month'] = predictions['date'].dt.month
    predictions['day'] = predictions['date'].dt.day
    predictions['dayofweek'] = predictions['date'].dt.dayofweek
    predictions['is_weekend'] = predictions['dayofweek'].isin([5, 6]).astype(int)
    predictions['day_name'] = predictions['date'].dt.day_name()
    predictions['month_name'] = predictions['date'].dt.month_name()
    
    return predictions
