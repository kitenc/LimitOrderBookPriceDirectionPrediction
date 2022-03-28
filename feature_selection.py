import pandas as pd
import numpy as np
from minepy import MINE
import random
import xgboost as xgb

from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


def feature_selection_data(full_data, labels, tk):
    full_data = full_data[(full_data['Time'] > (9.5+ 10 / 60) * 60 * 60) & (full_data['Time'] < (16 - 5/60) * 60 * 60)].reset_index(drop=True)
    X = full_data.iloc[:, :-len(labels)].drop(['OrderID','ticker'], axis=1)
    y = full_data.iloc[:, -tk]
    feature = pd.DataFrame([0]* full_data.shape[1], columns=full_data.iloc[:1000, :-len(labels)].drop(['OrderID','ticker'], axis=1).columns.values)
    feature = feature.T
    return X, y, feature



def feature_selection_xgb(X, y, feature):

    X_normal = normalize(X)
    # data_dmatrix = xgb.DMatrix(data=X,label=y)
    X_train, X_test, y_train, y_test = train_test_split(X_normal, y, test_size=0.2, random_state=123)

    xgb_base = xgb.XGBClassifier(colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

    xgb_base.fit(X_train, y_train)
    pred = xgb_base.predict(X_test)

    random_guess = np.array([random.choice([-1,0, 1]) for _ in range(len(y_test))])


    print("xgb precision:", precision_score(y_test, pred, average='weighted'))
    print("random guess precision:", precision_score(y_test, random_guess, average='weighted'))

    feature.columns=['xgb']

    return feature


def feature_selection_chi(X, y, feature):
    scaler = MinMaxScaler()
    X_minmax = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.2, random_state=123)
    chi_test = SelectKBest(score_func=chi2, k=4)
    chi_fit = chi_test.fit(X_train, y_train)

    feature['chi'] = chi_fit.scores_.reshape(-1,1)


def feature_selection_mie(X, y, feature):
    X_normal = normalize(X)
    sample_mine_length = 5000
    sample_mine = [np.random.randint(1, X_normal.shape[0]) for _ in range(1, sample_mine_length)]


    X_mine = X_normal[sample_mine,:]
    y_mine = y[sample_mine]

    mine_score = []
    for i in range(X_mine.shape[1]):
        m = MINE()
        m.compute_score(X_mine[:,i],y_mine)
        mine_score.append(m.mic())

    feature['mic'] = mine_score

    return feature


def feature_selection_ensemble(full_data, labels, tk=7):
    X, y, feature = feature_selection_data(full_data, labels, tk)
    feature = feature_selection_xgb(X, y, feature)
    feature = feature_selection_chi(X, y, feature)
    feature = feature_selection_mie(X, y, feature)
    methods = ['xgb','chi','mic']

    for column in methods:
        feature[ column + '_rank'] = feature[column].rank(axis=0, ascending=False)
        feature[ column + '_weight'] =  1- feature[column + '_rank'] / feature.shape[0]

    feature['total_arith_weight'] = ( feature['xgb_weight'] + feature['chi_weight'] + feature['mic_weight'] ) / 3
    feature['total_geo_weight'] = np.sqrt( feature['xgb_weight']**2 + feature['chi_weight']**2 + feature['mic_weight']**2 ) / 3

    return feature
