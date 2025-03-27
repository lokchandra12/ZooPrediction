import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def plot_historical_data(df, time_unit='daily'):
    """
    Plot historical visitor data in daily, weekly, or monthly view.
    
    Args:
        df (DataFrame): Processed zoo visitor data
        time_unit (str): Time unit for aggregation ('daily', 'weekly', or 'monthly')
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    plt.figure(figsize=(10, 6))
    
    if time_unit == 'daily':
        # Plot daily data
        plt.plot(df['date'], df['total_visitors'], marker='o', linestyle='-', alpha=0.5, label='Daily Visitors')
        plt.plot(df['date'], df['ma7_visitors'], linestyle='-', color='red', linewidth=2, label='7-Day Moving Avg')
        plt.title('Daily Zoo Visitors')
        
    elif time_unit == 'weekly':
        # Resample to weekly data
        agg_dict = {
            'total_visitors': 'sum',
            'adult_tickets': 'sum',
            'child_tickets': 'sum',
            'total_revenue': 'sum'
        }
        
        # Add additional ticket types if they exist
        if 'foreigner_tickets' in df.columns:
            agg_dict['foreigner_tickets'] = 'sum'
        if 'camera_tickets' in df.columns:
            agg_dict['camera_tickets'] = 'sum'
        if 'hend_camera_tickets' in df.columns:
            agg_dict['hend_camera_tickets'] = 'sum'
            
        weekly_data = df.set_index('date').resample('W').agg(agg_dict).reset_index()
        
        plt.plot(weekly_data['date'], weekly_data['total_visitors'], marker='o', linestyle='-', linewidth=2)
        plt.title('Weekly Zoo Visitors')
        
    elif time_unit == 'monthly':
        # Resample to monthly data
        agg_dict = {
            'total_visitors': 'sum',
            'adult_tickets': 'sum',
            'child_tickets': 'sum',
            'total_revenue': 'sum'
        }
        
        # Add additional ticket types if they exist
        if 'foreigner_tickets' in df.columns:
            agg_dict['foreigner_tickets'] = 'sum'
        if 'camera_tickets' in df.columns:
            agg_dict['camera_tickets'] = 'sum'
        if 'hend_camera_tickets' in df.columns:
            agg_dict['hend_camera_tickets'] = 'sum'
            
        monthly_data = df.set_index('date').resample('M').agg(agg_dict).reset_index()
        
        plt.plot(monthly_data['date'], monthly_data['total_visitors'], marker='o', linestyle='-', linewidth=2)
        plt.title('Monthly Zoo Visitors')
    
    plt.xlabel('Date')
    plt.ylabel('Number of Visitors')
    plt.grid(True, alpha=0.3)
    if time_unit == 'daily':
        plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_predictions(predictions_df):
    """
    Plot future attendance predictions.
    
    Args:
        predictions_df (DataFrame): Predicted attendance data
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot visitors
    ax1.plot(predictions_df['date'], predictions_df['total_visitors'], marker='o', linestyle='-', color='blue')
    ax1.set_title('Predicted Zoo Visitors')
    ax1.set_ylabel('Number of Visitors')
    ax1.grid(True, alpha=0.3)
    
    # Plot all ticket types as a stacked bar chart
    # Start with adult tickets at the bottom
    bottom = np.zeros(len(predictions_df))
    
    # Define colors for each ticket type
    colors = {
        'adult_tickets': 'green',
        'child_tickets': 'orange',
        'foreigner_tickets': 'blue',
        'camera_tickets': 'red',
        'hend_camera_tickets': 'purple'
    }
    
    # Define labels for each ticket type
    labels = {
        'adult_tickets': 'Adult Tickets',
        'child_tickets': 'Child Tickets',
        'foreigner_tickets': 'Foreigner Tickets',
        'camera_tickets': 'Camera Tickets',
        'hend_camera_tickets': 'H-END Camera Tickets'
    }
    
    # Add bars for each ticket type that exists in the data
    for ticket_type in ['adult_tickets', 'child_tickets', 'foreigner_tickets', 'camera_tickets', 'hend_camera_tickets']:
        if ticket_type in predictions_df.columns and predictions_df[ticket_type].sum() > 0:
            ax2.bar(predictions_df['date'], predictions_df[ticket_type], 
                   color=colors[ticket_type], alpha=0.7, label=labels[ticket_type], 
                   bottom=bottom)
            bottom += predictions_df[ticket_type].values
    ax2.set_title('Predicted Ticket Breakdown')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Tickets')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    return fig

def create_prediction_table(predictions_df):
    """
    Create a summary table of predictions by month.
    
    Args:
        predictions_df (DataFrame): Predicted attendance data
    
    Returns:
        DataFrame: Monthly summary of predictions
    """
    # Group predictions by month
    agg_dict = {
        'total_visitors': 'sum',
        'adult_tickets': 'sum',
        'child_tickets': 'sum',
        'total_revenue': 'sum'
    }
    
    # Add additional ticket types if they exist
    if 'foreigner_tickets' in predictions_df.columns:
        agg_dict['foreigner_tickets'] = 'sum'
    if 'camera_tickets' in predictions_df.columns:
        agg_dict['camera_tickets'] = 'sum'
    if 'hend_camera_tickets' in predictions_df.columns:
        agg_dict['hend_camera_tickets'] = 'sum'
        
    monthly_summary = predictions_df.groupby(['year', 'month', 'month_name']).agg(agg_dict).reset_index()
    
    # Format the summary table
    monthly_summary['month_year'] = monthly_summary['month_name'] + ' ' + monthly_summary['year'].astype(str)
    
    # Start with basic columns that must be present
    columns = ['month_year', 'total_visitors', 'adult_tickets', 'child_tickets']
    
    # Add additional ticket columns if they exist
    if 'foreigner_tickets' in monthly_summary.columns:
        columns.append('foreigner_tickets')
    if 'camera_tickets' in monthly_summary.columns:
        columns.append('camera_tickets')
    if 'hend_camera_tickets' in monthly_summary.columns:
        columns.append('hend_camera_tickets')
    
    # Always add total revenue at the end
    columns.append('total_revenue')
    
    # Reorder columns for display
    summary_table = monthly_summary[columns]
    
    # Create a mapping for renaming columns
    column_mapping = {
        'month_year': 'Month',
        'total_visitors': 'Total Visitors',
        'adult_tickets': 'Adult Tickets',
        'child_tickets': 'Child Tickets',
        'foreigner_tickets': 'Foreigner Tickets',
        'camera_tickets': 'Camera Tickets',
        'hend_camera_tickets': 'H-END Camera Tickets',
        'total_revenue': 'Total Revenue (₹)'
    }
    
    # Rename only columns that exist in the DataFrame
    rename_cols = {col: column_mapping[col] for col in columns if col in column_mapping}
    summary_table.columns = [rename_cols.get(col, col) for col in summary_table.columns]
    
    # Ensure integers for visitor counts
    for col in summary_table.columns:
        if col in ['Total Visitors', 'Adult Tickets', 'Child Tickets', 
                  'Foreigner Tickets', 'Camera Tickets', 'H-END Camera Tickets']:
            if col in summary_table.columns:
                summary_table[col] = summary_table[col].astype(int)
    
    # Format revenue as currency
    summary_table['Total Revenue (₹)'] = summary_table['Total Revenue (₹)'].round(2)
    
    return summary_table
