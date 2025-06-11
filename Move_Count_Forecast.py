import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import os
import pickle
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import nest_asyncio
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import random

# Apply nest_asyncio patch for Jupyter compatibility
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')  # Suppress Prophet logging

# Initialize FastAPI app
app = FastAPI(title="Move Forecast API", description="API to forecast move counts using Prophet models")

# PostgreSQL connection configuration
DATABASE_URL= os.environ.get("DATABASE_URL")

# Function to fetch data from PostgreSQL
def fetch_data(query):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        df = pd.read_sql_query(query, conn)
        if df.empty:
            logger.error(f"Query returned empty DataFrame: {query}")
            raise ValueError(f"No data found for query: {query}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data from PostgreSQL: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

# Function to initialize PostgreSQL (verify connection)
def init_db():
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        logger.info("PostgreSQL connection verified successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if conn is not None:
            conn.close()

# Initialize database
init_db()

# Function to save query to PostgreSQL
def save_query_to_db(input_date, branch, move_type, forecasted_count):
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        forecasted_count = int(forecasted_count)  # Convert to Python int
        cursor.execute("""
            INSERT INTO forecast_queries (input_date, branch, move_type, forecasted_count)
            VALUES (%s, %s, %s, %s)
        """, (input_date, branch, move_type, forecasted_count))
        conn.commit()
        logger.info(f"Saved query to database: {input_date}, {branch}, {move_type}")
    except Exception as e:
        logger.error(f"Error saving query to database: {str(e)}")
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

# Define input model for request validation
class ForecastInput(BaseModel):
    date: str  # e.g., "2025-06-11"
    branch: str  # e.g., "Columbus"
    move_type: Optional[str] = None  # e.g., "Local"

# Lists of varied comment phrases
CONSISTENT_PHRASES = [
    "Demand for {move_type} moves aligns closely with historical patterns (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "{move_type} move demand is in line with past trends (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "Expected {move_type} moves are consistent with historical data (historical avg {hist_avg:.1f}%, current {current:.1f}%)."
]
STRONGER_PHRASES = [
    "Demand for {move_type} moves is higher than historical trends (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "{move_type} move demand exceeds past patterns (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "Projected {move_type} moves show stronger demand than historical norms (historical avg {hist_avg:.1f}%, current {current:.1f}%)."
]
WEAKER_PHRASES = [
    "Demand for {move_type} moves is lower than historical trends (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "{move_type} move demand is below past trends (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "Expected {move_type} moves are weaker compared to historical data (historical avg {hist_avg:.1f}%, current {current:.1f}%)."
]
NO_MOVE_TYPE_PHRASE = "Forecast reflects total moves for the branch, with no move type specified."

# Summary comment phrases
SUMMARY_CONSISTENT_PHRASES = [
    "This period's {move_type} move demand in {branch} aligns with historical averages ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "{move_type} moves in {branch} for this period are consistent with past trends ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "The demand for {move_type} moves in {branch} this period matches historical patterns ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically)."
]
SUMMARY_STRONGER_PHRASES = [
    "This period's {move_type} move demand in {branch} is stronger than historical averages ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "{move_type} moves in {branch} show higher demand this period compared to past years ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "Demand for {move_type} moves in {branch} is elevated this period relative to historical trends ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically)."
]
SUMMARY_WEAKER_PHRASES = [
    "This period's {move_type} move demand in {branch} is lower than historical averages ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "{move_type} moves in {branch} are below historical trends for this period ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "Demand for {move_type} moves in {branch} is weaker this period compared to past years ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically)."
]
SUMMARY_NO_MOVE_TYPE = "This forecast reflects total moves for {branch} over the period, with no move type specified."

def forecast_move(input_date, input_branch, input_move_type=None, model_dir='prophet_models'):
    try:
        # Step 1: Load datasets from PostgreSQL
        historical_df = fetch_data('SELECT "Date", "Branch", "Count" FROM historical_df')
        move_df = fetch_data('SELECT "Date", "Branch", "MoveType", "Count" FROM move_df')
       
        # Validate column names
        required_historical_cols = {'Date', 'Branch', 'Count'}
        required_move_cols = {'Date', 'Branch', 'MoveType', 'Count'}
        if not required_historical_cols.issubset(historical_df.columns):
            raise ValueError(f"historical_df missing required columns: {required_historical_cols}")
        if not required_move_cols.issubset(move_df.columns):
            raise ValueError(f"move_df missing required columns: {required_move_cols}")
       
        # Ensure Date columns are in datetime format
        historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        move_df['Date'] = pd.to_datetime(move_df['Date'])
       
        # Convert input date to datetime
        try:
            input_date_dt = pd.to_datetime(input_date, format='%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD (e.g., '2025-06-11')")
       
        # Validate input date (up to July 31, 2025)
        if input_date_dt > pd.to_datetime('2025-07-31'):
            raise ValueError("Date must be on or before July 31, 2025")
       
        # Step 2: Aggregate historical_df by Date and Branch
        df_agg = historical_df.groupby(['Date', 'Branch'])['Count'].sum().reset_index()
       
        # Step 3: Define forecast periods
        train_end = pd.to_datetime('2023-12-31')
        forecast_end = pd.to_datetime('2025-07-31')
       
        # Step 4: Check if branch exists
        unique_branches = df_agg['Branch'].unique()
        if input_branch not in unique_branches:
            raise ValueError(f"Branch {input_branch} not found in data. Valid branches: {unique_branches.tolist()}")
       
        # Step 5: Validate move_type if provided
        if input_move_type is not None:
            valid_move_types = move_df['MoveType'].unique()
            if input_move_type not in valid_move_types:
                raise ValueError(f"Invalid MoveType. Valid MoveTypes: {valid_move_types.tolist()}")
       
        # Step 6: Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'prophet_model_{input_branch}.pkl')
       
        # Step 7: Load or train Prophet model for the branch
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            # Filter data for the branch
            branch_data = df_agg[df_agg['Branch'] == input_branch][['Date', 'Count']]
            branch_data = branch_data.rename(columns={'Date': 'ds', 'Count': 'y'})
           
            # Use data up to train_end for training
            train_data = branch_data[branch_data['ds'] <= train_end]
           
            # Skip if insufficient training data
            if len(train_data) < 2:
                raise ValueError(f"Insufficient training data for branch {input_branch}")
           
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=15.0,
                seasonality_mode='multiplicative'
            )
            model.fit(train_data)
           
            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
       
        # Step 8: Generate forecast for the 15-day window
        today = pd.to_datetime(datetime.now().date())  # Use current date dynamically
        days_from_today = (input_date_dt - today).days

        if days_from_today <= 7:
            # Input date is within 7 days from today, start from today
            start_date = today
            end_date = today + timedelta(days=14)  # 15-day window from today
        else:
            # Input date is more than 7 days from today, use ±7 days around input date
            start_date = input_date_dt - timedelta(days=7)
            end_date = input_date_dt + timedelta(days=7)

        # Cap end_date at forecast_end (July 31, 2025)
        if end_date > forecast_end:
            end_date = forecast_end
       
        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
       
        # Filter out dates before today
        forecast = forecast[forecast['ds'] >= today]
       
        # Extract forecast
        forecast = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Count'})
        forecast['Count'] = forecast['Count'].clip(lower=0).round().astype(int)
       
        # Step 9: Calculate historical percentage for MoveType (for input date)
        percentage = 100.0  # Default to 100% if no move_type
        if input_move_type is not None:
            percentages = []
            day = input_date_dt.day
            month = input_date_dt.month
           
            # Check previous 5 years (2020–2024)
            for year in range(2020, 2025):
                try:
                    historical_date = pd.to_datetime(f'{year}-{month:02d}-{day:02d}')
                except ValueError:
                    continue
               
                move_query = (move_df['Date'] == historical_date) & (move_df['Branch'] == input_branch) & (move_df['MoveType'] == input_move_type)
                move_counts = move_df[move_query]['Count'].sum()
               
                total_counts = historical_df[(historical_df['Date'] == historical_date) &
                                            (historical_df['Branch'] == input_branch)]['Count'].sum()
               
                if total_counts > 0:
                    percentage = (move_counts / total_counts) * 100
                    percentages.append(percentage)
           
            # Fallback if no data for the specific date
            if not percentages:
                move_query = (move_df['Branch'] == input_branch) & (move_df['Date'].dt.month == month) & (move_df['MoveType'] == input_move_type)
                move_counts = move_df[move_query]['Count'].sum()
                total_counts = historical_df[(historical_df['Branch'] == input_branch) &
                                            (historical_df['Date'].dt.month == month)]['Count'].sum()
               
                percentage = (move_counts / total_counts) * 100 if total_counts > 0 else 0
            else:
                percentage = sum(percentages) / len(percentages)
       
        # Step 10: Prepare output
        predicted_summary = []
        total_predicted_moves = 0
        total_branch_forecast = 0
       
        for _, row in forecast.iterrows():
            forecast_date = row['Date']
            branch_forecast = row['Count']  # Total branch forecast for the date
           
            # Calculate move_type-specific forecast
            final_forecast = (percentage / 100) * branch_forecast
            final_forecast = int(round(final_forecast))
           
            # Accumulate for summary comment
            total_predicted_moves += final_forecast
            total_branch_forecast += branch_forecast
           
            # Calculate comment by comparing historical percentage for this specific date
            comment = ""
            if input_move_type is not None:
                percentages = []
                day = forecast_date.day
                month = forecast_date.month
               
                # Check previous 5 years (2020–2024) for the same day and month
                for year in range(2020, 2025):
                    try:
                        historical_date = pd.to_datetime(f'{year}-{month:02d}-{day:02d}')
                    except ValueError:
                        continue
                   
                    move_query = (move_df['Date'] == historical_date) & (move_df['Branch'] == input_branch) & (move_df['MoveType'] == input_move_type)
                    move_counts = move_df[move_query]['Count'].sum()
                   
                    total_counts = historical_df[(historical_df['Date'] == historical_date) &
                                                (historical_df['Branch'] == input_branch)]['Count'].sum()
                   
                    if total_counts > 0:
                        hist_percentage = (move_counts / total_counts) * 100
                        percentages.append(hist_percentage)
               
                # Calculate historical average percentage
                historical_avg_percentage = sum(percentages) / len(percentages) if percentages else 0
               
                # Fallback if no data for the specific date
                if not percentages:
                    move_query = (move_df['Branch'] == input_branch) & (move_df['Date'].dt.month == month) & (move_df['MoveType'] == input_move_type)
                    move_counts = move_df[move_query]['Count'].sum()
                    total_counts = historical_df[(historical_df['Branch'] == input_branch) &
                                                (historical_df['Date'].dt.month == month)]['Count'].sum()
                   
                    historical_avg_percentage = (move_counts / total_counts) * 100 if total_counts > 0 else 0
               
                # Calculate implied percentage for the forecast
                implied_percentage = (final_forecast / branch_forecast * 100) if branch_forecast > 0 else 0
               
                # Compare with historical trend (using ±5% threshold)
                percentage_diff = implied_percentage - historical_avg_percentage
                if abs(percentage_diff) <= 5:
                    comment = random.choice(CONSISTENT_PHRASES).format(
                        move_type=input_move_type, hist_avg=historical_avg_percentage, current=implied_percentage
                    )
                elif percentage_diff > 5:
                    comment = random.choice(STRONGER_PHRASES).format(
                        move_type=input_move_type, hist_avg=historical_avg_percentage, current=implied_percentage
                    )
                else:
                    comment = random.choice(WEAKER_PHRASES).format(
                        move_type=input_move_type, hist_avg=historical_avg_percentage, current=implied_percentage
                    )
            else:
                comment = NO_MOVE_TYPE_PHRASE
           
            predicted_summary.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "predicted_moves": final_forecast,
                "comment": comment
            })
           
            # Save to database
            save_query_to_db(forecast_date, input_branch, input_move_type, final_forecast)
       
        # Calculate average daily moves
        average_daily_moves = int(round(total_predicted_moves / len(predicted_summary))) if predicted_summary else 0
       
        # Step 11: Calculate summary comment
        summary_comment = ""
        if input_move_type is not None:
            # Calculate current period's average percentage (2025)
            current_percentage = (total_predicted_moves / total_branch_forecast * 100) if total_branch_forecast > 0 else 0
           
            # Calculate historical percentage for the same period (2020–2024)
            historical_percentages = []
            for year in range(2020, 2025):
                period_percentages = []
                for forecast_date in pd.date_range(start=start_date, end=end_date, freq='D'):
                    if forecast_date < today:
                        continue
                    try:
                        historical_date = pd.to_datetime(f'{year}-{forecast_date.month:02d}-{forecast_date.day:02d}')
                    except ValueError:
                        continue
                   
                    move_query = (move_df['Date'] == historical_date) & (move_df['Branch'] == input_branch) & (move_df['MoveType'] == input_move_type)
                    move_counts = move_df[move_query]['Count'].sum()
                   
                    total_counts = historical_df[(historical_df['Date'] == historical_date) &
                                                (historical_df['Branch'] == input_branch)]['Count'].sum()
                   
                    if total_counts > 0:
                        period_percentage = (move_counts / total_counts) * 100
                        period_percentages.append(period_percentage)
               
                if period_percentages:
                    historical_percentages.append(sum(period_percentages) / len(period_percentages))
           
            # Calculate historical average percentage for the period
            historical_period_avg = sum(historical_percentages) / len(historical_percentages) if historical_percentages else 0
           
            # Fallback if no data for the specific dates
            if not historical_percentages:
                move_query = (move_df['Branch'] == input_branch) & (move_df['Date'].between(start_date, end_date)) & (move_df['MoveType'] == input_move_type)
                move_counts = move_df[move_query]['Count'].sum()
                total_counts = historical_df[(historical_df['Branch'] == input_branch) &
                                            (historical_df['Date'].between(start_date, end_date))]['Count'].sum()
               
                historical_period_avg = (move_counts / total_counts) * 100 if total_counts > 0 else 0
           
            # Compare current period with historical period
            period_percentage_diff = current_percentage - historical_period_avg
            if abs(period_percentage_diff) <= 5:
                summary_comment = random.choice(SUMMARY_CONSISTENT_PHRASES).format(
                    move_type=input_move_type, branch=input_branch, current_year=current_percentage, hist_avg=historical_period_avg
                )
            elif period_percentage_diff > 5:
                summary_comment = random.choice(SUMMARY_STRONGER_PHRASES).format(
                    move_type=input_move_type, branch=input_branch, current_year=current_percentage, hist_avg=historical_period_avg
                )
            else:
                summary_comment = random.choice(SUMMARY_WEAKER_PHRASES).format(
                    move_type=input_move_type, branch=input_branch, current_year=current_percentage, hist_avg=historical_period_avg
                )
        else:
            summary_comment = SUMMARY_NO_MOVE_TYPE.format(branch=input_branch)
       
        # Prepare the output structure
        result = {
            "branch": input_branch,
            "move_type": input_move_type,
            "forecast_window": {
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d')
            },
            "predicted_summary": predicted_summary,
            "total_predicted_moves": total_predicted_moves,
            "average_daily_moves": average_daily_moves,
            "summary_comment": summary_comment
        }
       
        return result
   
    except Exception as e:
        raise ValueError(str(e))

# API endpoint for forecasting

@app.get("/", response_model=dict)
@app.head("/")
async def root():
    return {"message": "Welcome to the Move Forecast API. Visit /docs for API documentation."}
   
@app.post("/forecast/")
async def forecast_endpoint(input_data: ForecastInput):
    try:
        logger.info(f"Received request: {input_data.dict()}")
        result = forecast_move(
            input_date=input_data.date,
            input_branch=input_data.branch,
            input_move_type=input_data.move_type
        )
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
