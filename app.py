import streamlit as st
import pandas as pd
import numpy as np
import io
import chardet
import openpyxl
import xlrd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from utils.data_processing import process_zoo_data, validate_csv_format
from utils.prediction import create_prediction_models, predict_future_attendance
from utils.visualization import plot_historical_data, plot_predictions, create_prediction_table

# Set page configuration
st.set_page_config(
    page_title="Zoo Visitor Prediction Tool",
    page_icon="🦁",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def main():
    st.title("Zoo Visitor and Revenue Prediction Tool")
    
    st.markdown("""
    This application helps zoo managers analyze historical visitor data and predict future attendance and revenue.
    
    **Instructions:**
    1. Upload a CSV file containing your zoo's visitor data
    2. The file should include: date, adult tickets sold, child tickets sold, adult ticket price, and child ticket price
    3. View the analysis of historical trends
    4. Use the prediction tool to forecast future attendance and revenue
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your zoo visitor data (CSV, Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Check file type
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type in ['xlsx', 'xls']:
                # Handle Excel files
                try:
                    # Read the Excel file directly without parsing dates
                    df = pd.read_excel(uploaded_file, parse_dates=False)
                    st.success(f"Excel file loaded successfully. Found {len(df)} rows and {len(df.columns)} columns.")
                    
                    # If we need to, we can directly send the dataframe to validation
                    content = df.to_csv(index=False)
                    
                    # Validate the data format
                    validation_result = validate_csv_format(content)
                    
                    if validation_result['valid']:
                        # Store the processed dataframe in session state
                        st.session_state.df = process_zoo_data(df)
                        
                        # Create prediction models
                        st.session_state.models = create_prediction_models(st.session_state.df)
                        
                        st.success("Data successfully processed!")
                    else:
                        st.error(f"Invalid data format: {validation_result['error']}")
                        st.markdown("""
                        **Expected data format:**
                        
                        | date/time stamp | adult_tickets | child_tickets | adult_price | child_price |
                        |------|---------------|---------------|-------------|-------------|
                        | YYYY-MM-DD or Timestamp | number | number | price | price |
                        """)
                        return
                        
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
                    return
            else:
                # Handle CSV files
                # Get file content as bytes
                file_bytes = uploaded_file.getvalue()
                
                # List of encodings to try
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 
                             'utf-16', 'utf-16-le', 'utf-16-be', 
                             'ascii', 'mac-roman', 'cp437']
                
                content = None
                successful_encoding = None
                
                # Try each encoding until one works
                for encoding in encodings:
                    try:
                        content = file_bytes.decode(encoding)
                        successful_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    st.error(f"Unable to decode the CSV file. We tried these encodings: {', '.join(encodings)}. Please check that your file is a valid CSV or try uploading an Excel file instead.")
                    
                    # Offer download option for a sample template
                    st.markdown("""
                    ### Sample Template
                    
                    You can download a sample CSV template below to see the expected format:
                    """)
                    
                    # Read the example file instead of hardcoding it
                    try:
                        with open('example_zoo_data.csv', 'r') as f:
                            sample_content = f.read()
                    except:
                        # Fallback to hardcoded sample that matches the client's data format
                        sample_content = """SR. NO\tContact Number\tWhatsapp Name\tVerification\tVerification Code\tBooking Date\tAdult Tickets\tChild Tickets\tForeigner Tickets\tCamera Tickets\tH-END Camera Tickets\tTotal Tickets\tTotal Amount Without Service Charge\tService Charge\tTotal Amount (INR)\tTicket Id\tTransaction Id\tPayment Status\tToken Verified By Username
1\t9190341234\tNeha\tTRUE\tD61DHWNFCA\t3/19/2025 9:37\t7\t1\t1\t2\t0\t11\t550\t16.5\t566.5\t09RBY314GV\tQ3Y1F7HBUVW1YD2P6G1G\tPending\trangeofficerrevenue@gmail.com
2\t9138041234\tRajesh\tTRUE\tF74W5E76DZ\t3/30/2025 9:06\t6\t3\t3\t0\t0\t12\t600\t18\t618\t7H56KVXJJ4\tA9IWVQ6RJBIJYU9L7CWO\tPending\trangeofficerrevenue@gmail.com
"""
                    st.download_button(
                        label="Download CSV Template",
                        data=sample_content,
                        file_name="zoo_template.csv",
                        mime="text/csv"
                    )
                    return
                
                st.success(f"File decoded successfully using {successful_encoding} encoding.")
                
                # Check if the content doesn't look like a CSV
                if ',' not in content[:1000] and ';' not in content[:1000] and '\t' not in content[:1000]:
                    st.warning("Your file might not be a properly formatted CSV. Please ensure it contains comma, semicolon, or tab-separated values.")
                
                # Validate the CSV format
                validation_result = validate_csv_format(content)
                
                if validation_result['valid']:
                    # Try to detect the delimiter (comma, semicolon, tab)
                    # For tab-delimited files like the user sample, we need to check more carefully
                    if '\t' in content[:1000]:
                        delimiter = '\t'
                        print("Detected tab-delimited file")
                    elif ';' in content[:1000]:
                        delimiter = ';'
                    else:
                        delimiter = ','  # Default delimiter
                    
                    data = io.StringIO(content)
                    # Don't specify parse_dates here - we'll handle date parsing in the validation function
                    df = pd.read_csv(data, delimiter=delimiter)
                    
                    # First pre-process the dataframe to ensure date column is created
                    processed_df = None
                    try:
                        # Lower case column names for consistency
                        df.columns = df.columns.str.lower()
                        
                        # Handle booking date conversion if available
                        if 'booking date' in df.columns:
                            # Format like "3/25/2025 7:52:00 AM"
                            try:
                                df['date'] = pd.to_datetime(df['booking date'], 
                                                          errors='coerce', 
                                                          format='%m/%d/%Y %I:%M:%S %p')
                                print(f"Converted dates using AM/PM format. Sample: {df['date'].head(3)}")
                            except Exception as e:
                                print(f"Error in specific AM/PM format: {e}")
                                # Check if the conversion worked
                            if df['date'].isnull().all():
                                # Try without seconds
                                try:
                                    df['date'] = pd.to_datetime(df['booking date'], 
                                                              errors='coerce', 
                                                              format='%m/%d/%Y %I:%M %p')
                                    print(f"Converted dates using AM/PM without seconds. Sample: {df['date'].head(3)}")
                                except Exception as e:
                                    print(f"Error in AM/PM without seconds format: {e}")
                            
                            # If still no valid dates, try 24-hour format
                            if df['date'].isnull().all():
                                try:
                                    df['date'] = pd.to_datetime(df['booking date'], 
                                                              errors='coerce', 
                                                              format='%m/%d/%Y %H:%M')
                                    print(f"Converted dates using 24-hour format. Sample: {df['date'].head(3)}")
                                except Exception as e:
                                    print(f"Error in 24-hour format: {e}")
                            
                            # Last fallback to pandas auto-detection
                            if df['date'].isnull().all():
                                df['date'] = pd.to_datetime(df['booking date'], errors='coerce')
                                print(f"Converted dates using pandas auto-detection. Sample: {df['date'].head(3)}")
                        
                        # Store the processed dataframe in session state
                        processed_df = process_zoo_data(df)
                        st.session_state.df = processed_df
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        import traceback
                        print(f"Full error details: {traceback.format_exc()}")
                        return
                    
                    # Create prediction models
                    st.session_state.models = create_prediction_models(st.session_state.df)
                    
                    st.success("Data successfully loaded and processed!")
                else:
                    st.error(f"Invalid CSV format: {validation_result['error']}")
                    st.markdown("""
                    **Expected CSV/Excel format:**
                    
                    Your file should contain these key columns (other columns are accepted too):
                    
                    - **Booking Date** (in format like "3/25/2025 9:37")
                    - **Adult Tickets** (number)
                    - **Child Tickets** (number)
                    - **Foreigner Tickets** (number, optional)
                    - **Camera Tickets** (number, optional)
                    - **H-END Camera Tickets** (number, optional)
                    - **Total Amount (INR)** (price in rupees, optional)
                    
                    We have detected that you're using a tab-delimited format which is fully supported.
                    """)
                    return
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            # Print more detailed exception info for debugging
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Error details: {traceback_str}")
            return
    
    # If data is loaded, display analysis and prediction interface
    if st.session_state.df is not None:
        display_data_analysis()
        display_prediction_interface()

def display_data_analysis():
    st.header("Historical Data Analysis")
    
    # Show data overview
    with st.expander("Data Overview", expanded=False):
        st.dataframe(st.session_state.df.head(10))
        
        st.markdown("### Data Summary")
        st.write(st.session_state.df.describe())
    
    # Plot historical data
    st.subheader("Visitor Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_daily = plot_historical_data(st.session_state.df, 'daily')
        st.pyplot(fig_daily)
    
    with col2:
        fig_weekly = plot_historical_data(st.session_state.df, 'weekly')
        st.pyplot(fig_weekly)
    
    # Find and display busiest and quietest days
    busiest_day = st.session_state.df.loc[st.session_state.df['total_visitors'].idxmax()]
    quietest_day = st.session_state.df.loc[st.session_state.df['total_visitors'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Busiest Day")
        st.markdown(f"**Date:** {busiest_day['date'].strftime('%Y-%m-%d')}")
        st.markdown(f"**Total Visitors:** {int(busiest_day['total_visitors'])}")
        st.markdown(f"**Adult Tickets:** {int(busiest_day['adult_tickets'])}")
        st.markdown(f"**Child Tickets:** {int(busiest_day['child_tickets'])}")
        
        # Add foreigner tickets if available
        if 'foreigner_tickets' in busiest_day and busiest_day['foreigner_tickets'] > 0:
            st.markdown(f"**Foreigner Tickets:** {int(busiest_day['foreigner_tickets'])}")
            
        # Add camera tickets if available
        if 'camera_tickets' in busiest_day and busiest_day['camera_tickets'] > 0:
            st.markdown(f"**Camera Tickets:** {int(busiest_day['camera_tickets'])}")
            
        # Add H-END camera tickets if available
        if 'hend_camera_tickets' in busiest_day and busiest_day['hend_camera_tickets'] > 0:
            st.markdown(f"**H-END Camera Tickets:** {int(busiest_day['hend_camera_tickets'])}")
            
        st.markdown(f"**Total Revenue:** ₹{busiest_day['total_revenue']:.2f}")
    
    with col2:
        st.markdown("### Quietest Day")
        st.markdown(f"**Date:** {quietest_day['date'].strftime('%Y-%m-%d')}")
        st.markdown(f"**Total Visitors:** {int(quietest_day['total_visitors'])}")
        st.markdown(f"**Adult Tickets:** {int(quietest_day['adult_tickets'])}")
        st.markdown(f"**Child Tickets:** {int(quietest_day['child_tickets'])}")
        
        # Add foreigner tickets if available
        if 'foreigner_tickets' in quietest_day and quietest_day['foreigner_tickets'] > 0:
            st.markdown(f"**Foreigner Tickets:** {int(quietest_day['foreigner_tickets'])}")
            
        # Add camera tickets if available
        if 'camera_tickets' in quietest_day and quietest_day['camera_tickets'] > 0:
            st.markdown(f"**Camera Tickets:** {int(quietest_day['camera_tickets'])}")
            
        # Add H-END camera tickets if available
        if 'hend_camera_tickets' in quietest_day and quietest_day['hend_camera_tickets'] > 0:
            st.markdown(f"**H-END Camera Tickets:** {int(quietest_day['hend_camera_tickets'])}")
            
        st.markdown(f"**Total Revenue:** ₹{quietest_day['total_revenue']:.2f}")
    
    # Display weekly patterns
    st.subheader("Weekly Patterns")
    
    # Group by day of week and calculate average
    day_of_week_avg = st.session_state.df.copy()
    day_of_week_avg['day_of_week'] = day_of_week_avg['date'].dt.day_name()
    day_of_week_avg = day_of_week_avg.groupby('day_of_week').agg({
        'total_visitors': 'mean',
        'adult_tickets': 'mean',
        'child_tickets': 'mean',
        'total_revenue': 'mean'
    }).reset_index()
    
    # Sort by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_avg['day_of_week'] = pd.Categorical(day_of_week_avg['day_of_week'], categories=day_order, ordered=True)
    day_of_week_avg = day_of_week_avg.sort_values('day_of_week')
    
    plt.figure(figsize=(10, 6))
    plt.bar(day_of_week_avg['day_of_week'], day_of_week_avg['total_visitors'])
    plt.title('Average Visitors by Day of Week')
    plt.ylabel('Average Number of Visitors')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

def display_prediction_interface():
    st.header("Attendance and Revenue Prediction")
    
    # Select prediction timeframe
    prediction_period = st.selectbox(
        "Select prediction timeframe",
        ["30 days", "3 months", "6 months", "1 year"]
    )
    
    # Map selection to number of days
    days_map = {
        "30 days": 30,
        "3 months": 90, 
        "6 months": 180,
        "1 year": 365
    }
    
    prediction_days = days_map[prediction_period]
    
    # Generate predictions if not already in session state or if period changed
    if st.session_state.predictions is None or len(st.session_state.predictions) != prediction_days:
        with st.spinner(f"Generating predictions for the next {prediction_period}..."):
            st.session_state.predictions = predict_future_attendance(
                st.session_state.df,
                st.session_state.models,
                prediction_days
            )
    
    # Display predictions
    st.subheader(f"Predicted Attendance for the Next {prediction_period}")
    
    # Plot predictions
    fig_pred = plot_predictions(st.session_state.predictions)
    st.pyplot(fig_pred)
    
    # Date selector for specific prediction
    latest_date = st.session_state.df['date'].max()
    min_date = latest_date + timedelta(days=1)
    max_date = latest_date + timedelta(days=prediction_days)
    
    selected_date = st.date_input(
        "Select a date to see detailed prediction",
        min_value=min_date,
        max_value=max_date,
        value=min_date
    )
    
    # Find the prediction for the selected date
    selected_pred = st.session_state.predictions[
        st.session_state.predictions['date'] == pd.Timestamp(selected_date)
    ]
    
    if not selected_pred.empty:
        st.subheader(f"Prediction for {selected_date}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Adult Tickets", int(selected_pred['adult_tickets'].values[0]))
        
        with col2:
            st.metric("Child Tickets", int(selected_pred['child_tickets'].values[0]))
        
        with col3:
            st.metric("Total Revenue", f"₹{selected_pred['total_revenue'].values[0]:.2f}")
    
    # Display prediction table
    st.subheader("Monthly Prediction Summary")
    pred_table = create_prediction_table(st.session_state.predictions)
    st.dataframe(pred_table)
    
    # Summary report
    st.header("Prediction Summary Report")
    
    total_visitors = st.session_state.predictions['total_visitors'].sum()
    total_revenue = st.session_state.predictions['total_revenue'].sum()
    avg_daily_visitors = st.session_state.predictions['total_visitors'].mean()
    avg_daily_revenue = st.session_state.predictions['total_revenue'].mean()
    
    st.markdown(f"""
    ### Key Findings
    
    - **Total Predicted Visitors:** {int(total_visitors)} over the next {prediction_period}
    - **Total Predicted Revenue:** ₹{total_revenue:.2f}
    - **Average Daily Visitors:** {int(avg_daily_visitors)}
    - **Average Daily Revenue:** ₹{avg_daily_revenue:.2f}
    
    ### Prediction Model Information
    
    This prediction tool uses a combination of time series forecasting methods:
    
    - Historical trends analysis
    - Seasonal pattern detection
    - Moving averages for smoothing
    - Prophet model for time series forecasting
    
    The model accounts for:
    - Day of week patterns
    - Monthly seasonality
    - Yearly seasonality (if sufficient data is available)
    
    ### How to Use This Tool
    
    1. **Upload fresh data regularly** to improve prediction accuracy
    2. **Select different timeframes** to plan for short and long-term scenarios
    3. **Check specific dates** for detailed daily predictions
    4. Use the **monthly summary** for budgeting and staffing decisions
    
    """)

if __name__ == "__main__":
    main()
