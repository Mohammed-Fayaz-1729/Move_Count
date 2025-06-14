import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

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

    branch = st.selectbox(
        "Select Branch",
        options=branch_options,
        index=0,
        help="Choose a branch location."
    )

    move_type = st.selectbox(
        "Select Move Type (Optional)",
        options=move_type_options,
        index=0,
        help="Choose a move type or leave empty."
    )

    submit_button = st.form_submit_button(label="Get Forecast")

# --- Caching the API call ---
@st.cache_data(show_spinner="Fetching forecast from API...")
def call_forecast_api_cached(date, branch, move_type):
    """
    Cached version of the API call.
    Caches results for each unique (date, branch, move_type) combination.
    """
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
        response = session.post(url, json=payload, timeout=30)  # Reduced timeout for free tier
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.json().get("detail", e.response.text)
        except Exception:
            error_detail = e.response.text
        return {"error": f"API Error (Status {e.response.status_code}): {error_detail}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network Error: Unable to connect to the API. {str(e)}"}

# Handle form submission
if submit_button:
    if not branch:
        st.error("Please select a valid Branch.")
    else:
        with st.spinner("Fetching forecast..."):
            result = call_forecast_api_cached(date, branch, move_type)
            if result:
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.json(result)
