# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 23:17:05 2022

@author: Saqib
"""
# Lime explainer
import lime
from lime import lime_tabular
from joblib import dump, load
import pickle
import numpy as np
import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split

# Scale and Normalize
from sklearn import preprocessing

class Model:
    def __init__(self):
        self.scaler = None
  
    def load_and_process(self, filename="./LEFT-140.0-FEATURES-AUG1.csv"):
        raw_df_all = pd.read_csv(filename, index_col=[0])
        # raw_df_all = raw_df_all.rename(columns={"RIGHT FOOT":"Overall Mean(Right)", "LEFT FOOT":"Overall Mean(Left)","IMC":"BMI"})
        # raw_df_all = raw_df_all.drop(columns=["Age (years)","Overall Mean(Right)"])
          
        return raw_df_all
    def data_load_split(self, df, train=0.7, val= 0.2, test=0.1):
      X = df[df.columns[:-1]].fillna(method="pad") # Features
      y = df.label.fillna(method="pad") # Target variable
    
      # Split train and test sets
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
      
      # Scale values 
      # X_train, X_test = scale_and_normalize(X,X_test)
      
      # Split train into val sets
      X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    
      return (X_train,y_train, X_val, y_val, X_test, y_test)
  
    def scale_and_normalize(self, X, X_val, X_test):
    
        x = X.values 
        xv = X_val.values
        xt = X_test.values
          
        cols = X.columns
        vcols =X_val.columns
        testcols = X_test.columns
          
        min_max_scaler = preprocessing.MinMaxScaler()
          
        scaler = min_max_scaler.fit(x)
        x_scaled = scaler.transform(x)
        x_val_scaled = scaler.transform(xv)
        x_test_scaled = scaler.transform(xt)
          
        X = pd.DataFrame(x_scaled, columns=cols)
        X_val = pd.DataFrame(x_val_scaled, columns = vcols)
        X_test = pd.DataFrame(x_test_scaled, columns=testcols)
        self.scaler = scaler
        return(X,X_val,X_test)  

    
    def load_classifier(self, filename):
        clf = load(f'./{filename}.joblib') 
        print(clf)
        return clf

    def get_scaler(self):
        df = self.load_and_process()
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_load_split(df) 
        X_train, X_val, X_test = self.scale_and_normalize(X_train, X_val, X_test)
        return self.scaler
        
    def generate_prediction(self, features):

        clf = self.load_classifier("xgb-LEFT-140.0-FEATURES-AUG1-clf")
        feat_n = ['contrast-2', 'mean-2', 'skewness-7', 'correlation-3', 'kurtosis-7', 'mean-7', 'ASM-7', 'correlation-7', 'kurtosis-2', 'variance-5', 'homogeneity-6', 'homogeneity-3', 'contrast-3', 'contrast-7', 'variance-7']

        df = self.load_and_process()
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_load_split(df) 
        X_train, X_val, X_test = self.scale_and_normalize(X_train, X_val, X_test)

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train[feat_n].values,
            feature_names = feat_n,
            kernel_width=5,
            class_names=['Control', 'Diabetes'],
            verbose=True,
            mode='classification'
        )
        #clf.get_booster().feature_names = np.array(X_train.columns.values)
        feat_names = feat_n
        clf.get_booster().feature_names = list(['f0','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14'])
        
        exp = explainer.explain_instance(
            data_row=features.iloc[0].values,
            predict_fn=clf.predict_proba
        )
    
        #exp.save_to_file('./feat-importance-a.htm', show_table=True, show_all=False)
        return exp.as_html(show_table=True, show_all=False)
        
        
        
