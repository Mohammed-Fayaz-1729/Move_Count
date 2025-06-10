import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime, timedelta
 
# Streamlit app configuration
st.set_page_config(page_title="Move Forecast App", page_icon="📈", layout="centered")
 
# Title and description
st.title("Move Forecast Application")
st.markdown("Enter a date, branch, and optional move type to forecast move counts using our API.")
 
# Hardcoded unique values
branch_options = [
    "", "Albuquerque", "Atlanta", "Atlanta 3", "Atlanta South", "Austin", "Austin South", "Baton Rouge",
    "Birmingham", "Boise", "Boston", "Boulder", "Boynton Beach", "Charleston", "Charlotte", "Charlotte South",
    "Chattanooga", "Cherry Hill", "Cheyenne", "Chicago", "Chicago 2", "Cincinnati", "Cincinnati 2", "Clarksville",
    "Cleveland", "Colorado Springs", "Columbia", "Columbus", "Columbus 2", "Connecticut", "Corpus Christi",
    "Dallas", "Dallas Downtown", "Dallas South", "Denton", "Denver", "Des Moines", "Detroit", "El Paso",
    "Fort Collins", "Fort Lauderdale", "Fort Worth", "Gainesville", "Greensboro", "Greenville", "Hartford",
    "Hilton Head", "Houston", "Houston North", "Houston South", "Indianapolis", "Indianapolis 2", "Interstate Services",
    "Jacksonville", "Jacksonville South", "Kansas City", "Kansas City Micro", "Knoxville", "Laramie", "Las Vegas",
    "Lexington", "Little Rock", "Littleton", "Louisville", "Maryland", "McKinney", "Memphis", "Mesa", "Miami",
    "Milwaukee", "Mobile", "Murfreesboro", "Myrtle Beach", "Naples", "Nashville", "New Orleans", "Newport News",
    "Oklahoma City", "Omaha", "Orlando", "Orlando 2", "Pensacola", "Philadelphia", "Phoenix", "Pittsburgh",
    "Portland", "Portland 2", "Raleigh", "Raleigh South", "Rhode Island", "Richmond", "Salem", "Salt Lake City",
    "Salt Lake City 2", "San Antonio", "San Antonio South", "Sanford", "Sarasota", "Savannah", "Seattle",
    "Seattle Micro", "St Louis", "Tampa", "Tucson", "Tulsa", "Tuscaloosa", "Twin Falls", "Virginia Beach",
    "Virginia North", "Waco"
]
 
move_type_options = [
    "", "Local", "Short Haul", "Labor Only", "INTERSTATE", "Long Distance", "ATS", "INTRA", "Corporate",
    "Unknown", "BOXES", "Short Haul Intrastate", "BackHaul", "Local Interstate", "Local Labor Only",
    "Long Distance Interstate", "Short Haul Interstate", "Linehaul", "Long Distance Backhaul", "Long Distance Linehaul"
]
 
# Input form
with st.form(key="forecast_form"):
    # Date input (restricted to >= current date, <= July 31, 2025)
    current_date = datetime.now().date()
    min_date = current_date
    max_date = datetime(2025, 7, 31).date()
    date = st.date_input(
        "Select Date",
        min_value=min_date,
        max_value=max_date,
        value=min_date,
        help=f"Choose a date between today ({current_date.strftime('%Y-%m-%d')}) and July 31, 2025."
    )
 
    # Branch dropdown
    branch = st.selectbox(
        "Select Branch",
        options=branch_options,
        index=0,
        help="Choose a branch location."
    )
 
    # Move Type dropdown
    move_type = st.selectbox(
        "Select Move Type (Optional)",
        options=move_type_options,
        index=0,
        help="Choose a move type or leave empty."
    )
 
    # Forecast button
    submit_button = st.form_submit_button(label="Get Forecast")
 
# Function to call the FastAPI endpoint with retry logic
def call_forecast_api(date, branch, move_type):
    url = "https://move-count-forecast-jfev.onrender.com/forecast/"
    payload = {
        "date": date.strftime("%Y-%m-%d"),
        "branch": branch,
        "move_type": move_type if move_type != "" else None
    }
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        response = session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.json().get("detail", e.response.text)
        except ValueError:
            error_detail = e.response.text
        st.error(f"API Error (Status {e.response.status_code}): {error_detail}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: Unable to connect to the API. {str(e)}")
        return None
 
# Handle form submission
if submit_button:
    if not branch:
        st.error("Please select a valid Branch.")
    else:
        with st.spinner("Fetching forecast..."):
            result = call_forecast_api(date, branch, move_type)
            if result:
                st.success("Forecast retrieved successfully!")
                # Display forecast summary
                st.subheader("Forecast Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Branch", result.get("branch"))
                    st.metric("Total Predicted Moves", result.get("total_predicted_moves"))
                with col2:
                    move_type_display = result.get("move_type") or "All Move Types"
                    st.metric("Move Type", move_type_display)
                    st.metric("Average Daily Moves", result.get("average_daily_moves"))
                # Display forecast window
                forecast_window = result.get("forecast_window", {})
                start_date = forecast_window.get("start_date", "N/A")
                end_date = forecast_window.get("end_date", "N/A")
                st.caption(f"Forecast period: {start_date} to {end_date}")
                # Display summary comment
                summary_comment = result.get("summary_comment", "")
                st.info(f"📊 **Summary Insight:** {summary_comment}")
                # Display daily forecasts in a table
                daily_data = result.get("predicted_summary", [])
                if daily_data:
                    st.subheader("Daily Forecast Details")
                    # Convert to DataFrame and format dates
                    df = pd.DataFrame(daily_data)
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%a, %b %d, %Y')
                    # Display as a table with formatted headers
                    st.dataframe(
                        df.rename(columns={
                            "date": "Date",
                            "predicted_moves": "Predicted Moves",
                            "comment": "Analysis"
                        }),
                        height=500
                    )
                    # Add download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Forecast Data",
                        data=csv,
                        file_name=f"move_forecast_{branch}_{date}.csv",
                        mime='text/csv',
                    )
                else:
                    st.warning("No daily forecast data available.")
                # Raw API response in expander
                with st.expander("View Raw API Response"):
                    st.json(result)
