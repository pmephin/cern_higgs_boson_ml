import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

##Feature Engineering

import warnings
warnings.simplefilter(action='ignore')

def forest_imputer(DF):
    
    P=RandomForestRegressor(n_estimators=100,n_jobs=4).fit(X=DF[DF.DER_mass_MMC!=-999][['DER_deltar_tau_lep','DER_mass_vis']],
       y=DF[DF.DER_mass_MMC!=-999].DER_mass_MMC).predict(DF[DF.DER_mass_MMC==-999][['DER_deltar_tau_lep','DER_mass_vis']])
    
    DF1=DF.copy()
    DF1.DER_mass_MMC[DF1.DER_mass_MMC==-999]=P
    return DF1

def linear_imputer(DF):
    
    lin_mod=LinearRegression().fit(X=DF[DF.DER_mass_MMC!=-999][['DER_deltar_tau_lep','DER_mass_vis']],
       y=DF[DF.DER_mass_MMC!=-999].DER_mass_MMC)
    
    #DF1=DF.copy()
    #DF1.DER_mass_MMC[DF1.DER_mass_MMC==-999]=P
    return lin_mod

def linear_impute_transform(df,lin_mod):
    P=lin_mod.predict(df[df.DER_mass_MMC==-999][['DER_deltar_tau_lep','DER_mass_vis']])
    df.DER_mass_MMC[df.DER_mass_MMC==-999]=P#._is_copy=P
    return df

def remove_phi_cols(df):
    phi_cols=[cols for cols in df.columns if 'phi' in cols.split('_')]
    df.drop(phi_cols[1:],axis=1,inplace=True)           ##removes all phi columns except centrality
    return df
    
def prepare_dataset(df):
    df.drop('EventId',axis=1,inplace=True)
    Weights=df.pop('Weight')
    y=df.pop('Label')
    
    #df=linear_imputer(df)
    y=pd.Series(np.where(y.values=='s',1,0))
    return(df,Weights,y)

def prepare_training_features(X,X_dep): ##transform X variables wrt to X_dep
    X=linear_impute_transform(X,linear_imputer(X_dep))
    X=SimpleImputer(missing_values=-999,strategy='median').fit(X_dep).transform(X)
    return X
    

### Modelling and evaluation

def ams_curve_tuning(X,y,weights,lr,n,rand_state,model):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=rand_state)
        
        lin_mod=linear_imputer(X_train)
        X_train=linear_impute_transform(X_train,lin_mod)#fit and transform
        X_test=linear_impute_transform(X_test,lin_mod)
        im=SimpleImputer(missing_values=-999,strategy='median')
        X_train=im.fit_transform(X_train)
        X_test=im.transform(X_test)
        
        model.fit(X_train,y_train)

        prediction_proba=model.predict_proba(X_test)[:,1]
        test_weights=weights.iloc[y_test.index]
        
        A=[]
        for t in np.arange(0.05,1,0.01):
            predictions=np.where(prediction_proba>t,1,0)
            s=np.dot(test_weights/0.3,((y_test==1)&(predictions==1)).astype('int64'))
                                                                                        ## Calculcating AMS for a threshold t
            b=np.dot(test_weights/0.3,((y_test==0)&(predictions==1)).astype('int64'))

            A.append(np.sqrt(2*(((s+b+10)*np.log(1+(s/(b+10))))-s)))
        return max(A)
    
def ams_curve(prediction_proba,y_test,test_weights,test_ratio):
    A=[]
    for t in np.arange(0.05,1,0.01):
        predictions=np.where(prediction_proba>t,1,0)
        s=np.dot(test_weights,((y_test==1)&(predictions==1)).astype('int64'))
                                                                                   
        b=np.dot(test_weights,((y_test==0)&(predictions==1)).astype('int64'))

        A.append(np.sqrt(2*(((s+b+10)*np.log(1+(s/(b+10))))-s)))
    A=np.array(A)
    Dat=np.stack((np.arange(0.05,1,0.01),A),axis=1)
    plt.plot(Dat[:,0],Dat[:,1])
    plt.title('AMS vs Decision Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('AMS')
    return Dat,max(A)
