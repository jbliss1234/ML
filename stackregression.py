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
from sklearn.metrics import explained_variance_score,r2_score,mean_absolute_error
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from sklearn.model_selection import KFold
from sklearn.base import clone

CONST_MAX_NN_ITER = 5000
CONST_MAX_TREE_DEPTH = 5
CONST_MIN_SPLIT_SAMPLES = 5 
CONST_NFOLDS = 6

def __get_list_learners (p_X):  
    def nn_model():
        "create a model."
        model = Sequential()
        model.add(Dense(400, input_dim = p_X.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(200, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dense(100, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adadelta')
        return model
    
    feature_columns = learn.infer_real_valued_columns_from_input(p_X) 

    list_learners = {
        #"XGB Regressor":                xgb.XGBRegressor(
        #                                max_depth=12, 
        #                                n_estimators=686, 
        #                                learning_rate=0.244947633008, 
        #                                silent=False, 
        #                                gamma=1.0, 
        #                                subsample=0.8500000000000001,
        #                                colsample_bytree=0.5,
        #                                colsample_bylevel=0.6756,
        #                                scale_pos_weight=1, 
        #                                min_child_weight=6,
        #                                max_delta_step=0,
        #                                seed=0)
        "KerasRegressor":               KerasRegressor(build_fn=nn_model, nb_epoch=50, verbose=0, batch_size=20,
                                                       validation_split=0.20, shuffle='batch')
        #"TF NN Regressor":              skflow.DNNRegressor(
        #                                hidden_units=[400, 200, 100, 5], 
        #                                feature_columns=feature_columns,
        #                                dropout=0.20),
    }

    return list_learners

def stack_regression_step1 (p_X, p_Y, p_MinError = None):   
    k_fold = KFold(CONST_NFOLDS)
    
    #Empty dictionary of learners that we will return 
    out_learners = {}    

    list_learners = __get_list_learners(p_X)
    
    for learner_name in list_learners:
        original_learner = list_learners[learner_name]
        
        counter = 0
        for k, (train, test) in enumerate(k_fold.split(p_X, p_Y)):
            counter = counter + 1
            learner = clone(original_learner)
            if learner_name == "TF Combined Neural Network" or learner_name == "TF NN Regressor":        
                learner.fit(p_X[train], p_Y[train], steps=CONST_MAX_NN_ITER, batch_size=32)            
            else:
                learner.fit(p_X[train], p_Y[train]) 
            
            if p_MinError is not None:
                #get metrics for this learner
                predictions, rmse, mae, evse, r2score = __get_prediction(learner_name, learner, p_X, p_Y)
                
                if evse >= p_MinError and r2score >= p_MinError:
                    out_learners["{} KFold# {}".format(learner_name, counter)] = learner
                else:
                    print("Discarding learner {} due to rmse {} and evse {} being les than threshold {}".format(learner_name, rmse, evse, p_MinError))
            else:                
                out_learners["{} KFold# {}".format(learner_name, counter)] = learner
        
    retVal = __get_predictions(out_learners, p_X, p_Y)

    return retVal, out_learners

def stack_regression_step2 (p_Learners, p_X):
    return __get_predictions(p_Learners, p_X, None);

def __get_predictions (p_Learners, p_X, p_Y = None):   
    retVal = pd.DataFrame()    
   
    for learner_name in list(p_Learners):
        learner = p_Learners[learner_name]       
        
        if p_Y is not None:
            predictions, rmse, mae, evse, r2score = __get_prediction(learner_name, learner, p_X, p_Y)
        else:
            predictions = __get_prediction(learner_name, learner, p_X, p_Y)

        predDF = pd.DataFrame(predictions)
        predDF.reset_index(inplace=True, drop=True)
        predDF.columns=[learner_name]        
        retVal = pd.concat([retVal, predDF], axis=1)        
        
    return retVal

def __get_prediction(p_LearnerName, p_Learner, p_X, p_Y = None):
    #predictions = list(regressor.predict(x_test, as_iterable=True))
    predictions = p_Learner.predict(p_X)
    
    if p_Y is not None:
        rmse = np.sqrt(metrics.mean_squared_error(predictions, p_Y))
        evse = explained_variance_score(p_Y, predictions)
        r2score = r2_score(p_Y, predictions)
        mae = mean_absolute_error(p_Y, predictions)
    
        if evse <= 0.50 and r2score <= 0.50:
            print("Consider discarding {} due to poor evse {} or r2score {}".format(p_LearnerName, evse, r2score))
        return predictions, rmse, mae, evse, r2score                  
    else:
        return predictions

def print_prediction_report(df_preds, y_actual, print_pred_df=False):    
    rmse, mae, evse, r2score = __get_combined_metrics(df_preds, y_actual)    
    print("RMSE: {}; MAE: {}; EVS: {}; R2score: {}".format(rmse, mae, evse, r2score))
    
    if print_pred_df == True:
        print(df_preds)
        
def __get_combined_metrics(df_preds, y_actual):
    df_preds['Mean Prediction'] = df_preds.mean(axis=1)
    df_yactual=pd.DataFrame(y_actual)
    df_yactual.reset_index(inplace=True, drop=True)
    df_yactual.columns=['Actual']
    df_preds = pd.concat([df_preds, df_yactual], axis=1)
    rmse = np.sqrt(metrics.mean_squared_error(df_preds['Mean Prediction'], y_actual))
    evse = explained_variance_score(y_actual, df_preds['Mean Prediction'])
    r2score = r2_score(y_actual, df_preds['Mean Prediction'])
    mae = mean_absolute_error(y_actual, df_preds['Mean Prediction'])
    
    return rmse, mae, evse, r2score
    