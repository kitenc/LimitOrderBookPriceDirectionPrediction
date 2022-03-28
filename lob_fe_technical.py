#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# Techinical Analysis

# Treat every second as a period ('trading day'), extact highest, lowest, and the close price of it


def feature_technical(Second,tech_horizons):
    columns = Second.columns.values
    Second = Second.merge(simple_moving_average(Second, tech_horizons), on='index', how='left')
    Second = Second.merge(wilder_moving_average(Second, tech_horizons), on='index', how='left')
    Second = Second.merge(average_true_range(Second, tech_horizons), on='index', how='left')
    Second = Second.merge(average_directional_index(Second, tech_horizons), on='index', how='left')
    Second = Second.merge(stochastic_oscillators(Second, tech_horizons), on='index', how='left')
    Second = Second.merge(awesome_oscillator(Second, tech_horizons), on='index', how='left')
    Second = Second.merge(relative_strength_index(Second, tech_horizons), on='index', how='left')
    Second = Second.merge(moving_average_convergence_divergence(Second, tech_horizons), on='index', how='left')
    Second = Second.merge(bollinger_bands(Second, tech_horizons), on='index', how='left')

    Second.drop(columns,axis=1, inplace=True)
    Second.fillna(method='bfill', inplace=True)
    return Second

# In[ ]:


# Simple Moving Average

def simple_moving_average(data, tech_horizons):
    columns= ['CloseSecond']

    try:
        Second = data[['index'] + columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 

    for horizon in tech_horizons:
        Second['SMA_' + str(horizon)] = Second['CloseSecond'].rolling(window=horizon).mean().fillna(method='bfill')
    Second['SMA_ratio'] = Second['SMA_'+str(tech_horizons[1])] / Second['SMA_'+str(tech_horizons[0])] 

    Second.drop('CloseSecond', axis=1, inplace=True)
    return Second

# Wilder’s Smoothing
# Put more weight on recent events
def Wilder(data, periods):
    start = np.where(~np.isnan(data))[0][0]
    Wilder = np.array([np.nan]*len(data))
    Wilder[start+periods-1] = data[start:(start+periods)].mean() 
    for i in range(start+periods,len(data)):
        Wilder[i] = (Wilder[i-1]*(periods-1) + data[i])/periods 
    return(Wilder)

def wilder_moving_average(data, tech_horizons):
    try:
        columns = ['SMA_'+str(tech_horizons[1]), 'SMA_'+str(tech_horizons[0])]
        Second = data[['index'] + columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 
        
    for horizon in tech_horizons:
        Second['Wilder_SMA_' + str(horizon)] = Wilder(Second['SMA_'+ str(horizon)], horizon)
    Second.fillna(method='bfill', inplace=True)
    Second['Wilder_SMA_ratio'] = Second['Wilder_SMA_'+str(tech_horizons[1])] / Second['Wilder_SMA_'+str(tech_horizons[0])]

    Second.drop(['SMA_'+str(tech_horizons[1]), 'SMA_'+str(tech_horizons[0])], axis=1, inplace=True)
    return Second


# In[ ]:


# Average True Range

def average_true_range(data, tech_horizons, opts = ['Max','Min','Close']):
    columns = [opt+'Second' for opt in opts]
    try:
        Second = data[['index'] + columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 

    for opt in opts:
        name = opt +'PrevSecond'
        columns.append(name)
        Second[name] = Second[opt+'Second'].shift(1).fillna(method='bfill')


    Second['TrueRange'] = np.maximum((Second['MaxSecond'] - Second['MinSecond']),
                                    np.maximum(abs(Second['ClosePrevSecond'] - Second['MinSecond']),
                                                abs(Second['MaxSecond'] - Second['ClosePrevSecond']))
                                    )

    for horizon in tech_horizons:
        Second['ATR_' + str(horizon)] = Wilder(Second['TrueRange'], horizon)
    Second.fillna(method='bfill', inplace=True)
    Second['ATR_ratio'] = Second['ATR_'+str(tech_horizons[1])] / Second['ATR_'+str(tech_horizons[0])]  

    Second.drop([opt+'Second' for opt in opts], axis=1, inplace=True)
    return Second
    


# In[ ]:


# Average Directional Index

def average_directional_index(data, tech_horizons):
    try:
        columns = ['MaxSecond', 'MinSecond', 'ATR_'+str(tech_horizons[1]), 'ATR_'+str(tech_horizons[0])]
        Second = data[['index']+ columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 
    
    for opt in ['Min', 'Max']:
        name = opt +'PrevSecond'
        columns.append(name)
        Second[name] = Second[opt+'Second'].shift(1)

    # Calculate direction movement
    Second['+DM'] = np.where(~np.isnan(Second['MaxPrevSecond']),
                            np.where((Second['MaxSecond'] > Second['MaxPrevSecond']) & 
                                        (((Second['MaxSecond'] - Second['MaxPrevSecond']) > (Second['MinPrevSecond'] - Second['MinSecond']))), 
                                Second['MaxSecond'] - Second['MaxPrevSecond'], 0),
                            np.nan)

    Second['-DM'] = np.where(~np.isnan(Second['MinPrevSecond']),
                            np.where((Second['MinSecond'] < Second['MinPrevSecond']) & 
                                        (((Second['MaxSecond'] - Second['MaxPrevSecond']) < (Second['MinPrevSecond'] - Second['MinSecond']))), 
                                Second['MinPrevSecond'] - Second['MinSecond'], 0),
                            np.nan)
    # Moving average of DM
    for horizon in tech_horizons:
        for sign in ['+', '-']:
            Second[sign + 'DM_' + str(horizon)] = Wilder(Second[sign + 'DM'], horizon)
            Second[sign + 'DI_' + str(horizon)] = Second[sign + 'DM_' + str(horizon)] / Second['ATR_' + str(horizon)] * 100

    for horizon in tech_horizons:
        name = 'DX_' + str(horizon)
        columns.append(name)
        Second[name] = (Second['+DI_' + str(horizon)] - Second['-DI_' + str(horizon)])/(Second['+DI_'+ str(horizon)] + Second['-DI_'+ str(horizon)]) * 100
        
        Second['ADX_'+ str(horizon)] = Wilder(Second['DX_'+ str(horizon)], horizon)

    for horizon in tech_horizons:
        name = 'Prev_ADX_'+ str(horizon)
        columns.append(name)
        Second[name] = Second['ADX_'+ str(horizon)].shift(1)
        Second['ADXR_' + str(horizon)] = (Second['ADX_' + str(horizon)] + Second['Prev_ADX_'+ str(horizon)]) / 2

    Second.drop(columns, axis=1, inplace=True)
    return Second
    


# In[ ]:


# Stochastic Oscillators

def stochastic_oscillators(data, tech_horizons):
    try:
        columns = ['MaxSecond', 'MinSecond', 'CloseSecond']
        Second = data[['index'] + columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 

    for horizon in tech_horizons:
        for direct in ['Max','Min']:
            name = direct + '_' +str(horizon)
            columns.append(name)
            Second[name] = Second[direct+'Second'].rolling(window=horizon).agg((direct.lower())).fillna(method='bfill')
        
        name = 'Stochastic_'+str(horizon)
        columns.append(name)
        Second[name] = 100 * Second['CloseSecond'] - Second['Min_'+str(horizon)] / (Second['Max_'+str(horizon)] - Second['CloseSecond'])
        Second['AvgStochastic_'+str(horizon)] = Second['Stochastic_'+str(horizon)].rolling(window=horizon).mean().fillna(method='bfill')

    Second['AvgStochastic_ratio'] = Second['AvgStochastic_'+str(tech_horizons[1])] / Second['AvgStochastic_'+str(tech_horizons[0])] 

    Second.drop(columns, axis=1, inplace=True)
    return Second


# Awesome oscillator
def awesome_oscillator(data, tech_horizons):
    try:
        columns = ['MaxSecond', 'MinSecond', 'SMA_'+str(tech_horizons[0]), 'SMA_'+str(tech_horizons[1])]
        Second = data[['index'] + columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 

    Second['AO'] = (Second['SMA_'+str(tech_horizons[0])] - Second['SMA_'+str(tech_horizons[1])]) * (Second['MaxSecond'] + Second['MinSecond']) / 2

    # Accelerator oscillator
    Second['AC'] = Second['AO'] * (1 - Second['SMA_'+str(tech_horizons[0])])

    Second.drop(columns, axis=1, inplace=True)
    return Second



# Relative Strength Index
def relative_strength_index(data, tech_horizons):
    try:
        columns = ['CloseSecond']
        Second = data[['index'] + columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 

    Second['CloseDiffToPrev'] = Second['CloseSecond'].diff()
    Second['CloseDiffUp'] = Second['CloseDiffToPrev']
    Second.loc[Second['CloseDiffUp'] < 0, 'CloseDiffUp'] = 0.0001

    Second['CloseDiffDown'] = Second['CloseDiffToPrev']
    Second.loc[Second['CloseDiffDown'] > 0, 'CloseDiffDown'] = 0.0001

    columns.extend(['CloseDiffToPrev', 'CloseDiffUp', 'CloseDiffDown'])

    for horizon in tech_horizons:
        for opt in ['Up', 'Down']:
            name = 'CloseDiffAvg' + opt + '_' + str(horizon)
            Second[name] = Second['CloseDiff' + opt].rolling(window=horizon).mean()
            columns.append(name)
        
        name = 'RS_'+str(horizon)
        columns.append(name)
        Second[name] = Second['CloseDiffAvgUp_' + str(horizon)] / Second['CloseDiffAvgDown_' + str(horizon)]
        
        Second['RSI_'+str(horizon)] = 100 - (100 / (1 + Second['RS_'+str(horizon)]))

    Second['RSI_ratio'] = Second['RSI_'+str(tech_horizons[0])] / Second['RSI_'+str(tech_horizons[1])]
    Second.drop(columns, axis=1, inplace=True)
    return Second


# In[ ]:


# Moving Average Convergence Divergence
def moving_average_convergence_divergence(data, tech_horizons):
    try:
        columns = ['CloseSecond']
        Second = data[['index'] + columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 

    for horizon in tech_horizons:
        Second['EWM_' + str(horizon)] = Second['CloseSecond'].ewm(span=horizon, adjust=False).mean()

    Second['MACD'] = Second['EWM_'+str(tech_horizons[1])] - Second['EWM_'+str(tech_horizons[0])]
    Second.drop(columns, axis=1, inplace=True)
    return Second


# In[ ]:


# Bollinger Bands

def bollinger_bands(data, tech_horizons):
    try:
        columns = ['CloseSecond']
        Second = data[['index'] + columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 


    for horizon in tech_horizons:
        for opt in ['mean', 'std']:
            name = 'Close'+opt+'_'+str(horizon)
            columns.append(name)
            Second[name] = Second['CloseSecond'].rolling(window=horizon).agg(opt)
        
        Second['UpperBBond_'+str(horizon)] = Second['Closemean_'+str(horizon)] + Second['Closestd_'+str(horizon)] * 2
        Second['LowerBBond_'+str(horizon)] = Second['Closemean_'+str(horizon)] - Second['Closestd_'+str(horizon)] * 2
    
    Second.drop(columns, axis=1, inplace=True)
    return Second


# In[ ]:

# Ichimoku clouds

def ichimoku_clouds(data):
    try:
        columns = ['MaxSecond', 'MinSecond']
        Second = data[['index'] + columns]
    except:
        print("Require columns not in the dataset: " , columns)
        return 

    for time in [9, 26, 52]:
        for opt in ['Max', 'Min']:
            name = opt +'_' + str(time)
            columns.append(name)
            Second[name] = Second[opt+'Second'].rolling(window=time).max()
    
    Second['ConversionLine'] = (Second['Max_9'] + Second['Min_9']) / 2
    Second['BaseLine'] = Second['Max_26'] + Second['Min_26']
    Second['LeadningSpanA'] = (Second['ConversionLine'] + Second['BaseLine']) / 2
    Second['LeadningSpanB'] = (Second['Max_52'] + Second['Min_52']) / 2

    Second.drop(columns, axis=1, inplace=True)
    return Second


# In[ ]:



# In[ ]:


