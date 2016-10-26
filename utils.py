import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

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