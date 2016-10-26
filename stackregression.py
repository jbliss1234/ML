#imports
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as skflow
from scipy.stats import zscore
from sklearn.cross_validation import KFold
from sklearn import metrics, preprocessing, datasets
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from tensorflow.contrib import learn
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet,PassiveAggressiveRegressor,LassoLarsCV,SGDRegressor
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score,r2_score
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from sklearn.model_selection import KFold

def stack_regression (p_X, p_Y):    
    CONST_MAX_NN_ITER = 500
    CONST_MAX_TREE_DEPTH = 5
    CONST_MIN_SPLIT_SAMPLES = 5
    CONST_NFOLDS = 10
    
    k_fold = KFold(3)
    
    def nn_model():
        "create a model."
        model = Sequential()
        model.add(Dense(400, input_dim = p_X.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(200, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adadelta')
        return model
    
    
    feature_columns = learn.infer_real_valued_columns_from_input(p_X) 
    out_learners = {
        #"SVR":                          SVR(kernel='rbf', C=1000, gamma='auto'),
        #"BayesianRidge":                BayesianRidge(),
        #"Gradient Boosting Regressor":  GradientBoostingRegressor(n_estimators=CONST_MAX_NN_ITER, max_depth=10,                       #                                  min_samples_split=2,learning_rate=0.1, random_state=np.random.RandomState(43)),
        #"TF Combined Neural Network":   skflow.DNNLinearCombinedRegressor(
        #                                dnn_hidden_units=[250, 125, 50, 5], 
        #                                dnn_feature_columns=feature_columns,
        #                                dnn_dropout=0.20,
        #                                linear_feature_columns=feature_columns),
        #"TF NN Regressor":              skflow.DNNRegressor(
        #                                hidden_units=[250, 125, 50, 5], 
        #                                feature_columns=feature_columns,
        #                                dropout=0.320),
        #"MLP Regressor":                MLPRegressor(shuffle=True, activation="relu", alpha=0.7, max_iter=CONST_MAX_NN_ITER,
        #                                verbose=False, early_stopping=True),
        #"Random Forest Regressor":      RandomForestRegressor(n_estimators=CONST_MAX_NN_ITER, max_depth=5,
        #                                min_samples_split=2),
        "XGB Regressor":                xgb.XGBRegressor(max_depth=10, n_estimators=CONST_MAX_NN_ITER, 
                                        learning_rate=0.1,nthread=10, silent=False, gamma=0.4,                           subsample=0.8,colsample_bytree=0.8,scale_pos_weight=1, seed=65),
        #"ElasticNet":                   ElasticNet(alpha=0.7, max_iter=CONST_MAX_NN_ITER),
        #"PassiveAggressiveRegressor":   PassiveAggressiveRegressor(C=0.7, n_iter=CONST_MAX_NN_ITER),
        #"LassoLarsCV":                  LassoLarsCV(precompute="auto", max_iter=CONST_MAX_NN_ITER),
        #"SGD Regressor":                SGDRegressor(),
        "KerasRegressor":               KerasRegressor(build_fn=nn_model, nb_epoch=10, verbose=1)
    }

    
    for learner_name in out_learners:
        learner = out_learners[learner_name]
        
        for k, (train, test) in enumerate(k_fold.split(p_X, p_Y)):
            if learner_name == "TF Combined Neural Network" or learner_name == "TF NN Regressor":        
                learner.fit(p_X[train], p_Y[train], steps=CONST_MAX_NN_ITER, batch_size=32)            
            else:
                learner.fit(p_X[train], p_Y[train])        
        
    retVal = stack_prediction(p_X, p_Y, out_learners, True)
        
    return retVal, out_learners

def stack_prediction (p_X, p_Y, p_Learners, p_isTraining=False):    
    retVal = pd.DataFrame()
    
    for learner_name in list(p_Learners):
        learner = p_Learners[learner_name]

        predictions, rmse,evse,r2score = get_prediction(p_X, p_Y, learner_name, learner)

        #if evse > 0.7 and r2score > 0.7:
        predDF = pd.DataFrame(predictions)
        predDF.reset_index(inplace=True, drop=True)
        predDF.columns=[learner_name]        
        retVal = pd.concat([retVal, predDF], axis=1)
        #elif p_isTraining == True:
        #    print("Discarding {}. EVSE={}; R2Score={}".format(learner_name, evse, r2score))
        #    p_Learners.pop(learner_name)           
        
    return retVal

def get_prediction(p_X, p_Y, p_LearnerName, p_Learner):
    #predictions = list(regressor.predict(x_test, as_iterable=True))
    predictions = list(p_Learner.predict(p_X))
    rmse = np.sqrt(metrics.mean_squared_error(predictions, p_Y))
    evse = explained_variance_score(p_Y, predictions)
    r2score = r2_score(p_Y, predictions)
    #print("Learner: {}; Score (RMSE): {}".format(p_LearnerName, rmse))
    #print("Explained variance score: {}".format(evse))
    #print("Coeff of determination: {}".format(r2score))
    return predictions, rmse, evse, r2score

def print_prediction_report(df_preds, y_actual, print_pred_df=False):
    df_preds['Mean Prediction'] = df_preds.mean(axis=1)
    df_yactual=pd.DataFrame(y_actual)
    df_yactual.reset_index(inplace=True, drop=True)
    df_yactual.columns=['Actual']
    df_preds = pd.concat([df_preds, df_yactual], axis=1)
    rmse = np.sqrt(metrics.mean_squared_error(df_preds['Mean Prediction'], y_actual))
    evse = explained_variance_score(y_actual, df_preds['Mean Prediction'])
    r2score = r2_score(y_actual, df_preds['Mean Prediction'])
    print("RMSE: {}; EVS:{}; R2score:{}".format(rmse, evse, r2score))
    
    if print_pred_df == True:
        print(df_preds)