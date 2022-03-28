# LimitOrderBookPriceDirectionPrediction

## Data
Lobster Limit Order Book data with event description
https://lobsterdata.com/info/DataSamples.php
Covers the whole trading time of a day, excludes the auction period


## Algorithm

#### Goal: 
Given a limit order book of one security and one specific time spot for past 200 events, predict the direction of security price movement in the next 1/5/10/50 events.

#### Label:
Price go down, price go up, price remain (in a small range), for different time horizons (decided freely)

## Feature Engineering
- Time sensitive
- Time insensitve
- Technical features:
    - wilder moving average
    - average true range
    - average direct index
    - stochastic/awesome oscillators
    - relative strength index
    - moving average convergence divergence
    - bollinger bands ... 

### Feature selection
Ensemble method 
    - cross entropy
    - maximum information
    - chi-squre test
    
## Deep learning model
Use the adjusted DeepLOB model introduced by https://arxiv.org/abs/1808.03668
To test whether feature engineering + the stat-to-art model could achieve a better performance with lower training and prediction cost

