from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, shapiro
import matplotlib.pyplot as plt
import requests
from time import sleep

# Function to fetch historical stock data from Yahoo Finance
def get_stock_data(symbol, start_date, end_date, retries=3):
    for _ in range(retries):
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date)['Close']
            return stock_data
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
            st.error(f"Network error fetching data for {symbol}: {e}. Retrying...")
            sleep(1)  # Wait for 1 second before retrying
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    st.error(f"Failed to fetch data for {symbol} after {retries} retries.")
    return None

# Rest of the code remains unchanged...

# Welcome window
def welcome_window():
    st.title('Welcome to the Stock Price Prediction App')
    st.write("""
    ### What is Multivariable Regression?
    Multivariable regression is a statistical technique that aims to model the relationship between a dependent variable and multiple independent variables. This technique allows us to understand how the dependent variable changes as the independent variables change.

    ### Why Use Multivariable Regression?
    By considering multiple predictors, multivariable regression provides a more comprehensive understanding of the factors influencing the dependent variable. This can lead to more accurate predictions and insights.

    ### What is Monte Carlo Forecasting?
    Monte Carlo forecasting is a statistical technique used to predict the future behavior of a variable by simulating a large number of random samples. This technique helps in understanding the range of possible outcomes and their probabilities by considering the uncertainty and variability in the input data.

    ### Recommended Values for the Model
    - **Significance Level:** Commonly used significance levels are 0.05, 0.01, and 0.001. A lower significance level means stricter criteria for determining significance.
    - **Desired Correlation:** Aim for a high correlation (close to 1) to ensure a strong relationship between the predictors and the dependent variable.
    
    ### How to Use This App
    1. Enter the stock symbol for the dependent variable (e.g., AAPL for Apple).
    2. Enter the stock symbols for the predictor variables.
    3. Select the start and end dates for the historical data.
    4. Choose the significance level and desired correlation.
    5. Click 'Fetch Data' to perform the regression analysis.
    6. Optionally, rerun the regression excluding non-significant variables.
    """)
    if st.button('Continue'):
        st.session_state.welcome_done = True

# Streamlit App
if 'welcome_done' not in st.session_state:
    welcome_window()
else:
    st.title('Stock Price Prediction')
    symbol_y = st.text_input('Enter Stock Symbol for Y (e.g., AAPL)', 'AAPL')
    start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('2024-01-01'))
    significance_level = st.selectbox('Choose level of significance', [0.05, 0.01, 0.001])
    desired_correlation = st.number_input('Desired correlation', min_value=0.1, max_value=1.0, value=0.8, step=0.01)

    # Select the number of predictor variables
    num_predictors = st.number_input('Number of predictor variables', min_value=1, max_value=5, value=1, step=1)
    predictor_symbols = []
    for i in range(num_predictors):
        predictor_symbols.append(st.text_input(f'Enter Stock Symbol for Predictor {i+1} (e.g., MSFT)', f'MSFT{i+1}'))

    # Rerun flag
    rerun = st.checkbox('Rerun without non-significant variables')

    if st.button('Fetch Data'):
        stock_data_y = get_stock_data(symbol_y, start_date, end_date)
        if stock_data_y is not None:
            # Fetch data for predictors
            predictors = []
            for symbol in predictor_symbols:
                stock_data_x = get_stock_data(symbol, start_date, end_date)
                if stock_data_x is not None:
                    predictors.append(stock_data_x)
            
            if len(predictors) == num_predictors:
                # Combine predictors into a single DataFrame
                predictors_df = pd.concat(predictors, axis=1)
                predictors_df.columns = predictor_symbols
                
                # Find common dates
                common_dates = stock_data_y.index
                for df in predictors:
                    common_dates = common_dates.intersection(df.index)
                
                # Filter stock data to include only common dates
                stock_data_y = stock_data_y.loc[common_dates]
                predictors_df = predictors_df.loc[common_dates]
                
                # Perform OLS regression
                model, y_pred = perform_ols_regression(predictors_df, stock_data_y)
                
                # Evaluate regression results
                residuals, mean_residuals, std_residuals, shapiro_stat, shapiro_pvalue, normality = evaluate_regression(stock_data_y, y_pred, significance_level)
                
                # Get model summary and statistics
                model_summary = model.summary()
                r_value = model.rsquared
                p_values = model.pvalues[1:]  # p-values for the predictors
                
                # Determine significance and correlation
                significance = ['Significant' if p < significance_level else 'Not Significant' for p in p_values]
                correlation = 'Strong correlation' if r_value > desired_correlation else 'Not enough correlation'
                
                if rerun:
                    # Filter out non-significant predictors
                    significant_predictors = [predictor for predictor, sig in zip(predictor_symbols, significance) if sig == 'Significant']
                    predictors_df = predictors_df[significant_predictors]
                    
                    # Perform OLS regression again
                    model, y_pred = perform_ols_regression(predictors_df, stock_data_y)
                    
                    # Evaluate regression results again
                    residuals, mean_residuals, std_residuals, shapiro_stat, shapiro_pvalue, normality = evaluate_regression(stock_data_y, y_pred, significance_level)

                # Perform Monte Carlo Forecasting
                num_simulations = 1000
                expected_value_mc = monte_carlo_forecast(stock_data_y, model, predictors_df, num_simulations)
                
                # Recommendations
                st.write('### Recommendations:')
                current_price = stock_data_y[-1]
                expected_price_regression = model.params[0] + sum(model.params[1:] * predictors_df.iloc[-1])
                
                if correlation != 'Strong correlation':
                    st.write(f"The correlation between predictors and {symbol_y} is not strong enough ({r_value:.2f}). It is recommended not to purchase the stock based on this analysis.")
                elif any(sig == 'Not Significant' for sig in significance):
                    st.write(f"Some predictors are not significant: {', '.join([predictor for predictor, sig in zip(predictor_symbols, significance) if sig == 'Not Significant'])}. It is recommended to rerun the analysis excluding non-significant predictors.")
                elif expected_price_regression > current_price and expected_value_mc > current_price:
                    st.write(f"The expected prices from both the regression model (${expected_price_regression:.2f}) and Monte Carlo forecasting (${expected_value_mc:.2f}) are higher than the current price (${current_price:.2f}). It is recommended to consider purchasing the stock as it is predicted to increase in value.")
                else:
                    st.write(f"Either the expected price from the regression model (${expected_price_regression:.2f}) or the expected value from Monte Carlo forecasting (${expected_value_mc:.2f}) is not higher than the current price (${current_price:.2f}). It is recommended to exercise caution before purchasing the stock.")

                # Display the results
                st.write('### Results:')
                st.write(f'Correlation between predictors and {symbol_y}: {correlation}')
                st.write(f'Significance of Predictors:')
                for predictor, sig in zip(predictor_symbols, significance):
                    st.write(f'{predictor}: {sig}')
                st.write(f'Current Price ({symbol_y}): {current_price}')
                st.write(f'Expected Price ({symbol_y}): {expected_price_regression:.2f}')
                st.write(f'Expected Value using Monte Carlo Forecasting: {expected_value_mc:.2f}')
                st.write(f'Residuals Distribution: {normality}')

                # Plot results
                plot_results(predictors_df, stock_data_y, y_pred, residuals, mean_residuals, std_residuals)

                # Regression data
                st.write(f'Model Summary:')
                st.text(model_summary)
                st.write(f'shapiro stat: {shapiro_stat}')
                st.write(f'shapiro p-value: {shapiro_pvalue}')

    # Add button to go back to welcome window
    if st.button('Go Back to Welcome'):
        st.session_state.welcome_done = False
        st.experimental_rerun()

