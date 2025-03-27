import pandas as pd
import numpy as np
from datetime import datetime
import io

def validate_csv_format(csv_content):
    """
    Validates if the CSV file has the required columns and format.
    
    Args:
        csv_content (str): CSV file content as string
    
    Returns:
        dict: Dictionary with validation result and error message if any
    """
    try:
        # Try to detect if it's an Excel file accidentally saved as CSV
        if b'\x50\x4b\x03\x04' in csv_content.encode('utf-8', errors='ignore')[:100] or \
           b'\xd0\xcf\x11\xe0' in csv_content.encode('utf-8', errors='ignore')[:100]:
            return {
                'valid': False,
                'error': "This appears to be an Excel file saved with a .csv extension. Please save it as an actual CSV file."
            }
            
        data = io.StringIO(csv_content)
        
        # Try to guess the delimiter
        delimiter = ','  # Default delimiter
        if ';' in csv_content[:1000]:
            delimiter = ';'
        elif '\t' in csv_content[:1000]:
            delimiter = '\t'
        
        # Try reading with the detected delimiter
        try:
            df = pd.read_csv(data, delimiter=delimiter)
        except Exception:
            # If fails, try pandas' automatic delimiter detection
            data.seek(0)
            try:
                # Try with Python's csv module first
                import csv
                dialect = csv.Sniffer().sniff(csv_content[:1000])
                delimiter = dialect.delimiter
                data.seek(0)
                df = pd.read_csv(data, delimiter=delimiter)
            except Exception as e:
                return {
                    'valid': False,
                    'error': f"Could not parse CSV: {str(e)}. Please check the file format and ensure it's properly formatted CSV."
                }
        
        # Clean column names (remove whitespace, lowercase)
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Check if the DataFrame is empty or has no columns
        if df.empty or len(df.columns) < 2:
            return {
                'valid': False,
                'error': "The file appears to be empty or improperly formatted. Please check that it contains valid CSV data."
            }
        
        # Display available columns - useful for debugging
        print(f"Available columns: {', '.join([str(col) for col in df.columns])}")
        
        # Check if this is the specific format with 'booking date', 'adult tickets', etc.
        zoo_format = False
        has_price_info = False  # Initialize the variable here
        
        if 'booking date' in df.columns and 'adult tickets' in df.columns and 'child tickets' in df.columns:
            zoo_format = True
            print("Detected zoo booking data format.")
            
        # Define different sets of required columns based on the format
        if zoo_format:
            # For the booking data format
            required_columns = ['booking date', 'adult tickets', 'child tickets']
            # We will calculate prices from the total amount
            if 'total amount (inr)' in df.columns or 'total amount without service charge' in df.columns:
                has_price_info = True
            else:
                has_price_info = False
        else:
            # Original expected format
            required_columns = ['date', 'adult_tickets', 'child_tickets', 'adult_price', 'child_price']
        
        column_mapping = {}
        missing_columns = []
        
        # Map column names based on semantic similarity
        for req_col in required_columns:
            matched = False
            
            # Direct match
            if req_col in df.columns:
                column_mapping[req_col] = req_col
                matched = True
                continue
            
            # Try to match based on substrings for the original format only
            if not zoo_format:
                for col in df.columns:
                    col_lower = col.lower()
                    
                    if req_col == 'date' and ('date' in col_lower or 'day' in col_lower or 'time' in col_lower):
                        column_mapping[col] = 'date'
                        matched = True
                        break
                    elif req_col == 'adult_tickets' and ('adult' in col_lower and ('ticket' in col_lower or 'visit' in col_lower or 'attendance' in col_lower)):
                        column_mapping[col] = 'adult_tickets'
                        matched = True
                        break
                    elif req_col == 'child_tickets' and ('child' in col_lower and ('ticket' in col_lower or 'visit' in col_lower or 'attendance' in col_lower)):
                        column_mapping[col] = 'child_tickets'
                        matched = True
                        break
                    elif req_col == 'adult_price' and ('adult' in col_lower and ('price' in col_lower or 'cost' in col_lower or 'fee' in col_lower or '$' in col_lower)):
                        column_mapping[col] = 'adult_price'
                        matched = True
                        break
                    elif req_col == 'child_price' and ('child' in col_lower and ('price' in col_lower or 'cost' in col_lower or 'fee' in col_lower or '$' in col_lower)):
                        column_mapping[col] = 'child_price'
                        matched = True
                        break
            
            if not matched:
                missing_columns.append(req_col)
        
        if missing_columns:
            return {
                'valid': False,
                'error': f"Missing required columns: {', '.join(missing_columns)}. Available columns: {', '.join([str(col) for col in df.columns])}"
            }
            
        # If we have the zoo booking format, prepare the data
        if zoo_format:
            # Check for timestamp column first, as it might be more reliable
            # Try handling both 'time stamp' (space) and 'timestamp' (no space) columns
            timestamp_col = None
            for col in df.columns:
                if col.lower().replace(' ', '') == 'timestamp':
                    timestamp_col = col
                    break
            
            if timestamp_col:
                # Convert Unix timestamp to datetime
                try:
                    # First make sure the column is treated as numeric
                    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
                    
                    # Debug info
                    print(f"Timestamp column values (first 5): {df[timestamp_col].head().tolist()}")
                    
                    # Get a valid sample timestamp (not NaN)
                    valid_timestamps = df[timestamp_col].dropna()
                    if not valid_timestamps.empty:
                        sample_ts = valid_timestamps.iloc[0]
                        print(f"Using sample timestamp: {sample_ts}")
                        
                        # Check if timestamps are in seconds format (10 digits starting with 17)
                        if sample_ts and str(int(sample_ts)).startswith('17'):
                            # Convert seconds to datetime
                            df['date'] = pd.to_datetime(df[timestamp_col], unit='s', errors='coerce')
                            print(f"Converted timestamps to dates using seconds. Sample: {df['date'].dropna().head(3)}")
                        else:
                            # Try milliseconds as fallback
                            df['date'] = pd.to_datetime(df[timestamp_col], unit='ms', errors='coerce')
                            print(f"Converted timestamps to dates using milliseconds. Sample: {df['date'].dropna().head(3)}")
                    else:
                        print("No valid timestamps found in the column")
                        
                except Exception as e:
                    print(f"Error converting timestamps: {str(e)}")
                    # Fallback to booking date if timestamp conversion fails
                    if 'booking date' in df.columns:
                        df['date'] = df['booking date']
                        print("Falling back to booking date column")
                    else:
                        print("No valid date column found")
            elif 'booking date' in df.columns:
                print("Found booking date column, trying to convert it")
                try:
                    # Try multiple date formats for the booking date
                    # First try a specific format: "3/25/2025 7:52:00 AM"
                    try:
                        df['date'] = pd.to_datetime(df['booking date'], 
                                                    errors='coerce', 
                                                    format='%m/%d/%Y %I:%M:%S %p')
                        if df['date'].isnull().all():
                            # Try format: "3/25/2025 7:52"
                            df['date'] = pd.to_datetime(df['booking date'], 
                                                        errors='coerce', 
                                                        format='%m/%d/%Y %H:%M')
                    except Exception as e:
                        print(f"Error in specific date format parsing: {e}")
                        # Fall back to pandas auto-detection
                        df['date'] = pd.to_datetime(df['booking date'], errors='coerce')
                    valid_dates = df['date'].dropna()
                    if not valid_dates.empty:
                        print(f"Successfully converted booking dates. Sample: {valid_dates.head(3)}")
                    else:
                        # Try without the specific format as a fallback
                        df['date'] = pd.to_datetime(df['booking date'], errors='coerce')
                        print("Using pandas auto date detection")
                except Exception as e:
                    print(f"Error converting booking date: {str(e)}")
                    # Still assign it even if conversion failed, we'll try general conversion later
                    df['date'] = df['booking date']
            
            # Map all ticket types
            if 'adult tickets' in df.columns:
                df['adult_tickets'] = df['adult tickets']
            if 'child tickets' in df.columns:
                df['child_tickets'] = df['child tickets']
            if 'foreigner tickets' in df.columns:
                df['foreigner_tickets'] = df['foreigner tickets']
            else:
                df['foreigner_tickets'] = 0
            if 'camera tickets' in df.columns:
                df['camera_tickets'] = df['camera tickets']
            else:
                df['camera_tickets'] = 0
            if 'h-end camera tickets' in df.columns:
                df['hend_camera_tickets'] = df['h-end camera tickets']
            else:
                df['hend_camera_tickets'] = 0
            
            # Calculate or set price info
            if has_price_info:
                # Estimate prices from total amount
                if 'total amount (inr)' in df.columns:
                    total_col = 'total amount (inr)'
                else:
                    total_col = 'total amount without service charge'
                
                # Use constant price for simplicity - can be refined
                if 'adult_price' not in df.columns:
                    # Default price estimate - can be adjusted
                    df['adult_price'] = 25.00
                if 'child_price' not in df.columns:
                    # Child price usually less than adult
                    df['child_price'] = 15.00
            else:
                # Set default prices if no price info is available
                df['adult_price'] = 25.00
                df['child_price'] = 15.00
                
            # Update required columns for standard processing
            required_columns = ['date', 'adult_tickets', 'child_tickets', 'adult_price', 'child_price']
        
        # Rename columns if needed
        if column_mapping:
            # Create a reverse mapping to rename to standard names
            reverse_mapping = {}
            for orig, mapped in column_mapping.items():
                reverse_mapping[orig] = mapped
            
            df = df.rename(columns=reverse_mapping)
        
        # Check date format
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Check if we have any null dates after conversion
            # Only consider it a problem if all dates are null
            if df['date'].isnull().all():
                return {
                    'valid': False,
                    'error': "All values in the 'date' column are invalid. Please ensure dates are in a standard format or valid timestamps."
                }
            
            # If we have some null dates, drop those rows
            if df['date'].isnull().any():
                null_count = df['date'].isnull().sum()
                total_count = len(df)
                print(f"Dropping {null_count} rows with invalid dates out of {total_count} total rows")
                df = df.dropna(subset=['date'])
                
            # Update the dataframe in memory
            df = df.copy()
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Error parsing dates: {str(e)}. The 'date' column should be in a valid date format (e.g., YYYY-MM-DD)."
            }
        
        # Check numeric columns
        numeric_columns = ['adult_tickets', 'child_tickets', 'adult_price', 'child_price']
        for col in numeric_columns:
            # Convert to numeric, with coercion to handle non-numeric values
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check if we have any null values after conversion
            if df[col].isnull().any():
                return {
                    'valid': False,
                    'error': f"Some values in the '{col}' column are not numeric. Please ensure all values are numbers."
                }
            
            # For ticket columns, values should be integers or convertible to integers
            if col in ['adult_tickets', 'child_tickets']:
                # Check if values are already integers
                if df[col].dtype == 'int64' or df[col].dtype == 'int32':
                    # Already integers, nothing to do
                    pass
                else:
                    # Convert to integers if possible and warn about it
                    try:
                        # Try to check if values are already whole numbers
                        has_decimals = False
                        for val in df[col]:
                            if isinstance(val, float) and not pd.isna(val):
                                if val != int(val):
                                    has_decimals = True
                                    break
                        
                        if has_decimals:
                            print(f"Warning: Some values in '{col}' are not integers. They will be rounded.")
                    except Exception as e:
                        print(f"Error checking integers: {e}")
        
        # Ensure no negative values
        for col in numeric_columns:
            if (df[col] < 0).any():
                return {
                    'valid': False,
                    'error': f"The '{col}' column contains negative values, which are not allowed."
                }
        
        return {'valid': True, 'error': None}
    
    except Exception as e:
        return {'valid': False, 'error': f"Validation error: {str(e)}"}

def process_zoo_data(df):
    """
    Process the zoo visitor data, calculate additional metrics, and prepare for analysis.
    
    Args:
        df (DataFrame): Raw zoo visitor data
    
    Returns:
        DataFrame: Processed data with additional calculated metrics
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure all column names are lowercase
    df.columns = df.columns.str.lower()
    
    # Handle the date column - first check if it exists
    if 'date' not in df.columns:
        # Try to create it from booking date if available
        if 'booking date' in df.columns:
            print("Creating date column from booking date")
            try:
                # Try multiple date formats
                df['date'] = pd.to_datetime(df['booking date'], errors='coerce', 
                                           format='%m/%d/%Y %I:%M:%S %p')
                # If all dates are null, try another format
                if df['date'].isnull().all():
                    df['date'] = pd.to_datetime(df['booking date'], errors='coerce', 
                                               format='%m/%d/%Y %H:%M')
                # If still all null, try general parsing
                if df['date'].isnull().all():
                    df['date'] = pd.to_datetime(df['booking date'], errors='coerce')
            except Exception as e:
                print(f"Error converting booking date: {e}")
                # Last resort - just copy the booking date column
                df['date'] = df['booking date']
        else:
            raise ValueError("No 'date' or 'booking date' column found in the data")
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with invalid dates
    invalid_dates = df['date'].isnull().sum()
    if invalid_dates > 0:
        print(f"Dropping {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Map ticket type columns if not already mapped
    # First check if we need to create them from original columns
    if 'adult_tickets' not in df.columns and 'adult tickets' in df.columns:
        df['adult_tickets'] = pd.to_numeric(df['adult tickets'], errors='coerce').fillna(0)
        
    if 'child_tickets' not in df.columns and 'child tickets' in df.columns:
        df['child_tickets'] = pd.to_numeric(df['child tickets'], errors='coerce').fillna(0)
        
    if 'foreigner_tickets' not in df.columns and 'foreigner tickets' in df.columns:
        df['foreigner_tickets'] = pd.to_numeric(df['foreigner tickets'], errors='coerce').fillna(0)
    elif 'foreigner_tickets' not in df.columns:
        df['foreigner_tickets'] = 0
        
    if 'camera_tickets' not in df.columns and 'camera tickets' in df.columns:
        df['camera_tickets'] = pd.to_numeric(df['camera tickets'], errors='coerce').fillna(0)
    elif 'camera_tickets' not in df.columns:
        df['camera_tickets'] = 0
        
    if 'hend_camera_tickets' not in df.columns and 'h-end camera tickets' in df.columns:
        df['hend_camera_tickets'] = pd.to_numeric(df['h-end camera tickets'], errors='coerce').fillna(0)
    elif 'hend_camera_tickets' not in df.columns:
        df['hend_camera_tickets'] = 0
    
    # Just in case we still don't have the main ticket columns, create them with zeros
    if 'adult_tickets' not in df.columns:
        print("Warning: No adult tickets column found, creating with zeros")
        df['adult_tickets'] = 0
    
    if 'child_tickets' not in df.columns:
        print("Warning: No child tickets column found, creating with zeros")
        df['child_tickets'] = 0
        
    # Calculate total visitors - include all ticket types
    df['total_visitors'] = df['adult_tickets'] + df['child_tickets']
    
    # Include foreigner tickets if available
    if 'foreigner_tickets' in df.columns:
        df['total_visitors'] += df['foreigner_tickets']
    
    # We don't count camera tickets in visitor count as they're not people
    
    # Calculate total revenue - this will be taken directly from the dataset
    # Set default prices if not present
    if 'adult_price' not in df.columns:
        df['adult_price'] = 25.00
    if 'child_price' not in df.columns:
        df['child_price'] = 15.00
    if 'foreigner_price' not in df.columns:
        df['foreigner_price'] = 50.00
    if 'camera_price' not in df.columns:
        df['camera_price'] = 10.00
    if 'hend_camera_price' not in df.columns:
        df['hend_camera_price'] = 20.00
        
    # Calculate revenue by ticket type
    df['adult_revenue'] = df['adult_tickets'] * df['adult_price']
    df['child_revenue'] = df['child_tickets'] * df['child_price']
    
    # Add revenues for additional ticket types
    if 'foreigner_tickets' in df.columns and df['foreigner_tickets'].sum() > 0:
        df['foreigner_revenue'] = df['foreigner_tickets'] * df['foreigner_price']
    else:
        df['foreigner_revenue'] = 0
        
    if 'camera_tickets' in df.columns and df['camera_tickets'].sum() > 0:
        df['camera_revenue'] = df['camera_tickets'] * df['camera_price']
    else:
        df['camera_revenue'] = 0
        
    if 'hend_camera_tickets' in df.columns and df['hend_camera_tickets'].sum() > 0:
        df['hend_camera_revenue'] = df['hend_camera_tickets'] * df['hend_camera_price']
    else:
        df['hend_camera_revenue'] = 0
    
    # Use actual total from data if available, otherwise calculate
    if 'total amount (inr)' in df.columns:
        df['total_revenue'] = df['total amount (inr)']
    else:
        df['total_revenue'] = df['adult_revenue'] + df['child_revenue'] + \
                              df['foreigner_revenue'] + df['camera_revenue'] + \
                              df['hend_camera_revenue']
    
    # Add date components for analysis
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['day_name'] = df['date'].dt.day_name()
    df['month_name'] = df['date'].dt.month_name()
    
    # Calculate 7-day moving averages
    df['ma7_visitors'] = df['total_visitors'].rolling(window=7, min_periods=1).mean()
    df['ma7_revenue'] = df['total_revenue'].rolling(window=7, min_periods=1).mean()
    
    # Calculate 30-day moving averages
    df['ma30_visitors'] = df['total_visitors'].rolling(window=30, min_periods=1).mean()
    df['ma30_revenue'] = df['total_revenue'].rolling(window=30, min_periods=1).mean()
    
    # Calculate percentage of adult vs child tickets
    df['adult_percentage'] = df['adult_tickets'] / df['total_visitors'] * 100
    df['child_percentage'] = df['child_tickets'] / df['total_visitors'] * 100
    
    return df
