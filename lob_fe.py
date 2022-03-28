import lob_fe_technical
import lob_fe_time


# feature generation
def feature_generation(df, nlevels, tech_horizons):
    df = lob_fe_time.feature_time(df, nlevels)
    df['TimeSecond'] = (df['Time']).astype(int) 

    Second = \
        df.groupby(['ticker','TimeSecond'])['Mid_Price'].agg(['min','max']).reset_index().merge(
            df.groupby(['ticker','TimeSecond'])['Mid_Price'].last().to_frame().reset_index(),
            on=['ticker','TimeSecond']
        )
    Second.columns=['ticker','TimeSecond','MinSecond','MaxSecond','CloseSecond']
    Second = Second.reset_index(drop=False)
    Second = Second.merge(lob_fe_technical.feature_technical(Second,tech_horizons).reset_index(drop=False),
                        on='index',
                        how='left'
                        )     
    Second.drop('index', axis=1, inplace=True)

    Second['PrevTimeSecond'] = Second['TimeSecond'].shift(1)
    Second.drop('TimeSecond', axis=1, inplace=True)
    Second.rename(columns={'PrevTimeSecond':'TimeSecond'}, inplace=True)  

    df = df.merge(Second, on=['TimeSecond','ticker'], how='left')  
    return df


# labeling

def generate_label(df, horizons, thres=0.002):
    labels = []
    for k in horizons:
        labels.append('label_'+str(k))
        df['Mid_Price_cumsum'] = df.groupby('ticker')['Mid_Price'].cumsum()
        df['Mid_Price_cumsum_next_'+str(k)] = df.groupby(['ticker'])['Mid_Price_cumsum'].shift(-k)

        df['PriceMoveNext_'+str(k)] = ((df['Mid_Price_cumsum_next_'+str(k)] - df['Mid_Price_cumsum']) / k - df['Mid_Price']) / df['Mid_Price']
        df['label_'+str(k)] = 1*(df['PriceMoveNext_'+str(k)] > thres)
        df.loc[df['PriceMoveNext_'+str(k)]<-thres, 'label_'+str(k)] = -1

        df.drop(['Mid_Price_cumsum_next_'+str(k), 'PriceMoveNext_'+str(k)], axis=1, inplace=True)
    
    df.drop(['Mid_Price_cumsum'], axis=1, inplace=True)
    
    return df, labels
