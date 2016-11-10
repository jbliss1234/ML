import os
import pandas as pd
import numpy as np
from scipy.special import logit
from sklearn import preprocessing
from random import randint
from sklearn.cross_validation import train_test_split
from sklearn import decomposition
import matplotlib.pyplot as plt

#define common functions
# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)    
def encode_text_dummy(df,name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name,x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    
#Encode text values for a list of cols
def encode_text_dummy_list(df,names):    
    for name in names:
        encode_text_dummy(df, name)
    
# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).    
def encode_text_index(df,name): 
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

def encode_text_index_list(df,names): 
    for name in names:
        encode_text_index(df, name)
        
def encode_text_index_all(df): 
    for name in df:
        encode_text_index(df)
        
# Encode a numeric column as log    
def encode_numeric_log(df,name): 
    df[name] = np.log10(df[name] + 1.01)
    
# Encode numeric as zscores for a list of columns  
def encode_numeric_log_list(df,names):
    for name in names:
        encode_numeric_log(df, name)
        
def encode_numeric_log_all(df): 
    for name in df:
        encode_numeric_log(df, name)
        
# Encode a numeric column as log    
def encode_numeric_logit(df,name): 
    df[name] = logit(1.0 + df[name])
    
# Encode numeric as zscores for a list of columns  
def encode_numeric_logit_list(df,names):
    for name in names:
        encode_numeric_logit(df, name)
        
def encode_numeric_logit_all(df): 
    for name in df:
        encode_numeric_logit(df, name)
                
# Encode a numeric column as zscores    
def encode_numeric_zscore(df,name,mean=None,sd=None):
    if mean is None:
        mean = df[name].mean()
        
    if sd is None:
        sd = df[name].std()
        
    df[name] = (df[name]-mean)/sd
    
# Encode numeric as zscores for a list of columns  
def encode_numeric_zscore_list(df,names,mean=None,sd=None):
    for name in names:
        encode_numeric_zscore(df, name, mean, sd)
        
# Encode numeric as zscores for a list of columns  
def encode_numeric_zscore_all(df, mean=None,sd=None):
    for name in df:
        encode_numeric_zscore(df, name, mean, sd)
    
# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df,target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    print(target_type)
    
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        return df.as_matrix(result).astype(np.float32),df.as_matrix([target]).astype(np.int32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32),df.as_matrix([target]).astype(np.float32)
    
def to_xyV2(df,target):
    dfy = df[target]
    df.drop(target, axis=1, inplace=True)    
    return df, dfy

def get_allstate_train_valid_test_testids(validfrac, shift, verbose=False):
    train = pd.read_csv('./data/allstate/train.csv')
    if verbose == True:
        print("Train shape is: {}".format(train.shape))
    
    train["type"] = "train"
    #shuffle train
    train = train.reindex(np.random.permutation(train.index))
    train.reset_index(inplace=True, drop=True)   
    
    test = pd.read_csv('./data/allstate/test.csv')
    if verbose == True:
        print("Test shape is: {}".format(test.shape))
        
    testids = test["id"]
    test["type"] = "test"
    test["loss"] = np.nan
    
    joined = pd.concat([train, test], axis=0) #stacks them vertically
    joined.drop("id", axis=1, inplace=True)
    encode_text_dummy_list(joined, ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18', 'cat19', 'cat20', 'cat21', 'cat22', 'cat23', 'cat24', 'cat25', 'cat26', 'cat27', 'cat28', 'cat29', 'cat30', 'cat31', 'cat32', 'cat33', 'cat34', 'cat35', 'cat36', 'cat37', 'cat38', 'cat39', 'cat40', 'cat41', 'cat42', 'cat43', 'cat44', 'cat45', 'cat46', 'cat47', 'cat48', 'cat49', 'cat50', 'cat51', 'cat52', 'cat53', 'cat54', 'cat55', 'cat56', 'cat57', 'cat58', 'cat59', 'cat60', 'cat61', 'cat62', 'cat63', 'cat64', 'cat65', 'cat66', 'cat67', 'cat68', 'cat69', 'cat70', 'cat71', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76', 'cat77', 'cat78', 'cat79', 'cat80', 'cat81', 'cat82', 'cat83', 'cat84', 'cat85', 'cat86', 'cat87', 'cat88', 'cat89', 'cat90', 'cat91', 'cat92', 'cat93', 'cat94', 'cat95', 'cat96', 'cat97', 'cat98', 'cat99', 'cat100', 'cat101', 'cat102', 'cat103', 'cat104', 'cat105', 'cat106', 'cat107', 'cat108', 'cat109', 'cat110', 'cat111', 'cat112', 'cat113', 'cat114', 'cat115', 'cat116'])
    encode_numeric_zscore_list(joined, ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14'])
    joined["loss"] = np.log(joined["loss"] + shift)

    train = joined[joined.type == "train"]
    test = joined[joined.type == "test"]
    train.drop("type", axis=1, inplace=True)
    test.drop("type", axis=1, inplace=True)
    
    valid = train.sample(frac=validfrac)
    train = train[~train.isin(valid)].dropna()
    
    #reset indexs before returning
    train.reset_index(inplace=True, drop=True)   
    valid.reset_index(inplace=True, drop=True)
    
    if verbose == True:
        print("Final Train shape is: {}".format(train.shape))
        print("Final Valid shape is: {}".format(valid.shape))
        print("Final Test shape is: {}".format(test.shape))
    
    return pd.DataFrame(train), pd.DataFrame(valid), pd.DataFrame(test), pd.DataFrame(testids)

def chart_regression(pred,y):
    t = pd.DataFrame({'pred' : pred.flatten(), 'y' : y.flatten()})
    t.sort_values(by=['y'],inplace=True)

    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()