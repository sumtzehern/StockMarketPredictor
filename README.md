# Stock Market Predictor using Random Forest Classifier

## Overview
This project involves building a stock market predictor using a Random Forest Classifier. The goal is to predict whether the stock market will go up or down based on historical data.

## Steps

### 1. Data Collection
- Utilized the Yahoo Finance API, Pandas, and os to collect historical data for the S&P 500 index (`^GSPC`).
- Data was collected from the beginning of available data to the present.

### 2. Data Cleaning and Preparation
- Cleaned the data by removing unnecessary columns like "Dividends" and "Stock Splits."
- Created a binary target variable ("Target") to check if the next day's closing price would be greater than the current day's closing price.
- Filtered the data to include only records after the year 1990.

### 3. Model Building
- Used a RandomForestClassifier to build the predictive model.
- Selected features like "Close," "Volume," "Open," "High," and "Low" as predictors.
- Trained the model on a portion of the data and tested its precision on a separate test set.

### 4. Backtesting
- Implemented a backtesting system to evaluate the model's performance over time.
- Conducted backtests at regular intervals, predicting short-term market movements.

### 5. Feature Engineering
- Introduced new features such as rolling averages and trends with different time horizons.
- Enhanced the model by incorporating additional relevant information.

## Results
- Achieved a precision score of approximately 55% in the initial model.
- Further enhanced precision to 57% by incorporating additional features.

## Suggestions for Improvement

1. **Extend Data Collection:**
   - Test the model on a more extensive dataset to evaluate its performance under various market conditions.

2. **Fine-Tune Model:**
   - Experiment with different parameters for the RandomForestClassifier to improve accuracy.
   - Adjust the threshold for predictions to achieve a better balance between precision and recall.

3. **Incorporate External Factors:**
   - Include external factors such as news sentiment, interest rates, and key events in the market.
   - Consider incorporating data from key stocks or sectors, especially those with potential correlation to the S&P 500.

4. **Increase Data Resolution:**
   - Explore higher-resolution data (e.g., intraday) for more accurate predictions.

5. **Market Open Timing:**
   - Account for the fact that the S&P 500 only trades during U.S. market hours. Consider aligning data with other index before open times.

6. **Evaluate Co-relations:**
   - Explore correlations with other indices or market-related variables.

7. **Feature Engineering:**
   - Continue experimenting with additional features and their impact on model performance.

## Fine Tuning
| n_estimators | min_samples_split | precision |
|--------------|-------------------|-----------|
| 100          | 100               | 0.5510    |
| 1000         | 500               | 0.6368    |


## Conclusion
This project provides a foundation for building a stock market predictor, and continuous refinement can lead to more accurate predictions. Experimentation with different features, models, and external factors will contribute to the development of a robust predictive tool.

