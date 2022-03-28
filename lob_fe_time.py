import pandas as pd

def feature_time(df, nlevels):
    df = df.sort_values(['Time'], ascending=True)

    # set market order
    df['NewType'] = df['Type']
    df.loc[(df['Type']==4)&(((df['Price']==df['Ask_Price_1'])&(df['TradeDirection']==1))|((df['Price']==df['Bid_Price_1'])&(df['TradeDirection']==-1))), 'NewType'] = 6
    # spread
    df['Spread'] = df['Ask_Price_1'] - df['Bid_Price_1']
    # mid-price
    df['Mid_Price'] = (df['Ask_Price_1'] - df['Bid_Price_1']) / 2


    df = feature_base_stats(df, nlevels)
    df = feature_price_difference(df, nlevels)
    df = feature_deviation(df, nlevels)

    return df





#--------------Time-Insensitive-----------------#
def feature_base_stats(df, nlevels):
    # price & volume mean
    opts = ['mean', 'std']
    for opt in opts:
        df['Ask_Price_'+ opt] = df[['Ask_Price_' + str(i) for i in range(1, nlevels+1)]].agg(opt, axis=1)
        df['Bid_Price_'+ opt] = df[['Bid_Price_' + str(i) for i in range(1, nlevels+1)]].agg(opt, axis=1)
        df['Ask_Size_'+ opt] = df[['Ask_Size_' + str(i) for i in range(1, nlevels+1)]].agg(opt, axis=1)
        df['Bid_Size_'+ opt] = df[['Bid_Size_' + str(i) for i in range(1, nlevels+1)]].agg(opt, axis=1)


    # accumulated differences
    df['Accumulated_Price_diff'] = df[['Ask_Price_' + str(i) for i in range(1, nlevels+1)]].mean(axis=1) - \
                                    df[['Bid_Price_' + str(i) for i in range(1, nlevels+1)]].mean(axis=1)
    df['Accumulated_Size_diff'] = df[['Ask_Size_' + str(i) for i in range(1, nlevels+1)]].mean(axis=1) - \
                                    df[['Bid_Size_' + str(i) for i in range(1, nlevels+1)]].mean(axis=1)

    return df

def feature_price_difference(df, nlevels):
    # price difference
    for i in range(2, nlevels+1):
        df["Ask_level_" + str(i) + "_diff_to_best"] = df['Ask_Price_' + str(i)] - df['Ask_Price_1']
        df["Bid_level_" + str(i) + "_diff_to_best"] = df['Bid_Price_' + str(i)] - df['Bid_Price_1']
        df["Ask_level_"+ str(i) + "_diff_to_previous"] = df['Ask_Price_' + str(i)] - df['Ask_Price_'+str(i-1)]
        df["Bid_level_" + str(i) + "_diff_to_previous"] = df['Bid_Price_' + str(i)] - df['Bid_Price_'+str(i-1)]
    
    return df

#--------------Time-Sensitive-----------------#

# price & volume deviation

def feature_deviation(data, nlevels):
    TimeIndex = data.groupby(['Time'])['Time'].mean().to_frame()
    TimeIndex['PreviousTime'] = TimeIndex['Time'].shift(1).fillna(method='bfill')
    PreList = ['Ask_Price_'+str(i) for i in range(1, nlevels+1)] + ['Bid_Price_'+str(i) for i in range(1, nlevels+1)] + \
        ['Ask_Size_'+str(i) for i in range(1, nlevels+1)] + ['Bid_Size_'+str(i) for i in range(1, nlevels+1)] 
    PreInfo = data.groupby(['Time'])[['Time'] + PreList].last().reset_index(drop=True)
    PreInfo.columns=['PreviousTime']+['Previous_' + PreList[i] for i in range(len(PreList))]
    PreInfo = PreInfo.merge(TimeIndex, on='PreviousTime', how='right').fillna(method='bfill')
    
    df = data.merge(PreInfo, left_on='Time', right_on='Time', how='left').drop('key_0', axis=1)
    df.PreviousTime.fillna(method='bfill').fillna(method='ffill')

    
    for i in range(1, nlevels):
        df['Deviation_Ask_Price_' +str(i)] = ((df['Ask_Price_'+str(i)] - df['Previous_Ask_Price_' + str(i)]) / (df['Time'] - df['PreviousTime'])).fillna(method='ffill')
        df['Deviation_Bid_Price_' +str(i)] = ((df['Bid_Price_'+str(i)] - df['Previous_Bid_Price_' + str(i)]) / (df['Time'] - df['PreviousTime'])).fillna(method='ffill')
        df['Deviation_Ask_Size_' +str(i)] = ((df['Ask_Size_'+str(i)] - df['Previous_Ask_Size_' + str(i)]) / (df['Time'] - df['PreviousTime'])).fillna(method='ffill')
        df['Deviation_Bid_Size_' +str(i)] = ((df['Bid_Size_'+str(i)] - df['Previous_Bid_Size_' + str(i)]) / (df['Time'] - df['PreviousTime'])).fillna(method='ffill')


    df.drop(['Previous_' + PreList[i] for i in range(len(PreList))], axis=1, inplace=True)
    df = df.fillna(method='bfill')
    df = df.fillna(df.mean, axis=1)

    return df
    


# average intensity per type