#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the yahooFinance API, pandas and os

import yfinance as yf
import pandas as pd
import os


# In[2]:


sp500 = yf.Ticker("^GSPC")


# In[3]:


sp500 = sp500.history(period="max")


# In[4]:


sp500


# In[5]:


sp500.index


# In[6]:


# Plot the data
sp500.plot.line(y="Close", use_index=True)


# In[7]:


# More for individual stock

del sp500["Dividends"]
del sp500["Stock Splits"]


# In[8]:


sp500["Tomorrow"] = sp500["Close"].shift(-1)


# In[9]:


sp500


# In[10]:


# set target to check if the tomorrow price will be greater than ystd closing price 

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


# In[11]:


sp500


# In[12]:


# Having historical is good, but too old data might not be too useful for predicting( marketshift etc)

# we clean data to show only after 1990
sp500 = sp500.loc["1990-01-01":].copy()


# In[13]:


sp500


# In[14]:


# RandomForestClassifier Reduces overfitting by averaging multiple decision trees
# Pick up non-linear relationship
# If you can find linear you gonna be rich!

from sklearn.ensemble import RandomForestClassifier 

# n_estimators: might be good with higher value
# min_samples_split: protect from overfitting, higher less accurate but also less overfitt
# random_state: we would run the same random data

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"] # if we use Tomorrow,Target we are will know the future which is no going to happen in real world
model.fit(train[predictors], train["Target"])

              


# In[15]:


from sklearn.metrics import precision_score

preds = model.predict(test[predictors])


# In[16]:


# Turn numpy array into panda series

preds = pd.Series(preds, index=test.index)


# In[17]:


preds


# In[18]:


precision_score(test["Target"], preds)

# we got 55% which is alright


# In[19]:


combined = pd.concat([test["Target"], preds], axis=1)


# In[20]:


# Orange: prediction
# Blue: actual

combined.plot()


# In[21]:


# Building a Backtesting System

'''
1. Fitting the model using training predictor
2. Generate prediction
3. Combine model into series
4. Combining all 
'''

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[22]:


# Trading around 250 per year

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)
        


# In[23]:


predictions = backtest(sp500, model, predictors)


# In[24]:


predictions["Predictions"].value_counts()


# In[25]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[26]:


predictions["Target"].value_counts() / predictions.shape[0]


# In[27]:


'''
2 days, last trading week, 3 month, last year, 4 years
'''

horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]


# In[28]:


sp500 = sp500.dropna() # drop the missing column


# In[29]:


sp500


# In[30]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[31]:


'''
1. Fitting the model using training predictor
2. Generate prediction
3. Combine model into series
4. Combining all 
'''

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    # Custom treshold
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    
    return combined


# In[32]:


predictions = backtest(sp500, model, new_predictors)


# In[33]:


predictions["Predictions"].value_counts()


# In[34]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[35]:


predictions["Target"].value_counts() / predictions.shape[0]


# predictions
# 
