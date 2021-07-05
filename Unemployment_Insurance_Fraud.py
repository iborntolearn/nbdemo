#!/usr/bin/env python
# coding: utf-8

# # <font color='green'>Model Builder Template</font> 
# ## Overview
# * 			This template is used for seamless integration with CDMS serving layer by preparing `custom_model.py` and `metadata.json` files.
# * 			When this template is opened, a folder with name same as Model Code was created in the local. 
# * 			The cells with code containing <i>import statements</i> or <i>function definitions</i> will be extracted to the `custom_model.py`.
# * 			Please refer to the Model Onboarding User Guide to get the detailed working of this template.

# ### Note: Rename your notebook file to some meaningful model before doing 			any model development. Changing notebook name after onboarding can make your code behave erroneously.

# In[1]:


initialize_metadata()


# ### Model Meta data Information
# * 			Please enter a unique model code, model name (e.g. Customer Churn, Propensity to Default, CLTV etc) and Model description.
# * 			Please ensure the 'Create Workspace' button is clicked.
# * 			For more information, please refer to the User Guide.

<<<<<<< HEAD
# In[2]:
=======
# In[6]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


enter_metadata_information(1)


# ### Import Packages
# * 			The following cell can be used to import all the required python packages.
# * 			Please import the `gr` package to use the pre-defined models and functionalities. 			For more information, please refer to the User Guide.

<<<<<<< HEAD
# In[3]:
=======
# In[7]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81



#imports 
import os, sys, pickle;
from datetime import datetime
from typing import Dict
#For address comparision
from fuzzywuzzy import fuzz

import numpy as np 
import pandas as pd
# To suppress SettingWithCopyWarning in the code
pd.options.mode.chained_assignment = None
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, stats
from imblearn.over_sampling import SMOTE, ADASYN 
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
    roc_auc_score,precision_recall_curve, f1_score, accuracy_score, auc)
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.feature_selection import RFE, SelectKBest, chi2, SelectFromModel
from sklearn.svm import LinearSVC
from functools import reduce

from sklearn.preprocessing import MinMaxScaler


from category_encoders import MEstimateEncoder
from sklearn.model_selection import (train_test_split,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV,cross_val_score,RepeatedKFold)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import traceback

import os
import pickle

from statsmodels.stats.outliers_influence import variance_inflation_factor

import gc
#model explanability import

import gzip

import pickle
import shap
import fsspec
from io import StringIO

import logging
import platform


<<<<<<< HEAD
# In[4]:
=======
# In[8]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81



log_level = logging.INFO
if platform.system() == "Windows":
    log_level = logging.DEBUG
logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('./fraud_detection.log')
        ]
    )
logger = logging.getLogger("cdms_pipeline")


<<<<<<< HEAD
# In[16]:


print("this is a test")


=======
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81
# ### Save Model 
# * 			The following method can be used to save the trained model in the model folder structure created. 
# * 			The method parameter `trained_model` accepts the estimator(trained model) that is to be stored as model `pickle` file. 
#  			The method parameter `filename` accepts a `string` containing the pickle file name.

<<<<<<< HEAD
# In[5]:
=======
# In[9]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


def save_model(trained_model, file_name):    
    model_pickle_path = None
    try:
        model_pickle_path = os.path.abspath(os.path.join(__file__, os.pardir, file_name))
    except Exception:
        model_pickle_path = os.path.join(os.path.abspath(''),file_name)
    logger.info(f"Saving to location {model_pickle_path}.")
    with open(model_pickle_path, 'wb') as pickle_out:
        pickle.dump(trained_model, pickle_out)


# ### Load Model 
# * 			The following method can be used to load the saved model from the model folder structure. 
# * 			The method parameter `filename` accepts a `string` containing the model pickle file name that is to be loaded.

<<<<<<< HEAD
# In[6]:
=======
# In[10]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


def load_model(file_name):    
    model = None
    model_pickle_path = None
    try:
        model_pickle_path = os.path.abspath(os.path.join(__file__, os.pardir, file_name))
    except Exception:
        model_pickle_path = os.path.join(os.path.abspath(''),file_name)
    logger.info(f"Loading model from location {model_pickle_path}.")
    with open(model_pickle_path, 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    return model


<<<<<<< HEAD
# In[7]:
=======
# In[11]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81



def clean_data(df_x, drop_cols_list):
    logger.info("clean_data")
    present_cols = [col for col in drop_cols_list if col in df_x.columns]
    df_x = df_x.drop(present_cols,axis = 1)
    return df_x

def calculate_age(born):
    today = datetime.today() 
    try:  
        birthday = born.replace(year = today.year) 

    # raised when birth date is February 29 
    # and the current year is not a leap year 
    except ValueError:  
        birthday = born.replace(year = today.year, 
                  month = born.month + 1, day = 1) 

    if birthday > today: 
        return today.year - born.year - 1
    else: 
        return today.year - born.year 


def is_ambigious_workexp(work_exp, age, work_start_age):
    if work_exp <= (age*12 - work_start_age):
        return 0
    else:
        return 1

def match_column(text1,text2,threshold):
    match_ratio = fuzz.token_sort_ratio(text1, text2)
    if match_ratio >= threshold:
        return 0
    else:
        return 1
    
def is_card_expired(exp_date, file_date):
    if exp_date is pd.NaT:
        return -1
    if exp_date < file_date:
        return 1
    else:
        return 0

def is_banking_fraud(payment_mode, res_mail_address_match):
    if payment_mode=='D' and res_mail_address_match==0:
        return 1
    else:
        return 0

def invalid_dob(age):
    if age < 16 or age>80:
        return 1
    else:
        return 0    

#Outlier functions
def q1_quantile(x):
    q25 = x.quantile(0.25)
    if (q25 < 0):
        return 0
    else:
        return q25


def q2_quantile(x):
    q50 = x.quantile(0.50)
    if (q50 < 0):
        return 0
    else:
        return q50

def q3_quantile(x):
    q75 = x.quantile(0.75)
    if (q75 < 0):
        return 0
    else:
        return q75

def upper_count_limit(q1_quantile, q3_quantile):
    iqr_count = q3_quantile - q1_quantile
    upper_limit = q3_quantile + (1.5*iqr_count)
    return upper_limit

def rename_keys(rename_dict, feature_important):
    logger.info("Rename keys")
    logger.debug(rename_dict)
    logger.debug(feature_important)
    new_dict = {}
    for k , v in feature_important.items():
        if rename_dict.get(k):
            new_dict[rename_dict.get(k)] = v
        else:
            new_dict[k] = v
    return new_dict    


<<<<<<< HEAD
# In[8]:
=======
# In[12]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


class Preprocessing():
    def __init__(self, file_system, input_dict, training=True):
        logger.info("Preprocessing initialization")

        self.fs = file_system
        self.input_dict = input_dict
        self.input_file_path = input_dict["file_path"]
        self.is_train = training
        
        self.fraud_indicator_cols = ['STATE_ID_THEFT','ID_EXPIRED','IS_ID_MISSING','IS_ID_NOT_LOCAL','IS_ID_STATE_NOT_MATCHING_MAIL',                            'IS_ID_STATE_NOT_MATCHING_RES','ACC_NUMBER_THEFT','IS_ACC_MISSING','EMAIL_THEFT','PHONE_THEFT',                            'LIVING_TOGETHER','CORRESPONDING_TOGETHER','NOT_SAME_MAIL_RES_ADDR','IS_SKILL_PREFERENCE_NOT_MATCHING',                            'IS_MAIL_STATE_NOT_LOCAL','IS_suspicious_DOB','INVALID_DOB','IS_WORK_EXP_AMBIGIOUS','IS_EMPLOYER_MISSING',                            'HAS_FRAUD_HISTORY']
        self.cols_dtype_dict = {"CLAIMANT_ID":str, "CLAIMANT_ID_x":str, "CLAIM_APPLICATION_ID":str, "CLAIMANT_ID_y":str}
        self.raw_per_df = self._read_csv_folder(os.path.join(self.input_file_path, self.input_dict["raw_per_df"]), low_memory=False, converters=self.cols_dtype_dict)
        self.interim_df = pd.DataFrame()
        # Check if folder is present and folder is not empty
        interim_df_path = os.path.join(self.input_file_path, self.input_dict["interim_df"])
        if self.fs.isdir(interim_df_path) and len(self.fs.ls(interim_df_path))>0:
            self.interim_df = self._read_csv_folder(interim_df_path, low_memory=False, converters=self.cols_dtype_dict)
        # During training, interim dataframe if present will be concated to raw_per_df.
        # During prediction, interim dataframe if present will be concated to raw_per_df_history.
        if self.is_train and not self.interim_df.empty:
            logger.info("Merging claimnat personal details interim data during training.")
            self.raw_per_df = pd.concat([self.raw_per_df, self.interim_df], sort=True).reset_index(drop=True)
        skill_df_path = os.path.join(self.input_file_path,  self.input_dict["skill_df"])
        self.skill_df = self._read_csv_folder(skill_df_path, low_memory=False, converters=self.cols_dtype_dict)
        employment_df_path = os.path.join(self.input_file_path, self.input_dict["employment_df"])
        self.employment_df = self._read_csv_folder(employment_df_path, low_memory=False, converters=self.cols_dtype_dict)
        employer_df_path = os.path.join(self.input_file_path, self.input_dict["employer_df"])
        self.employer_df = self._read_csv_folder(employer_df_path, low_memory=False, converters=self.cols_dtype_dict)
        claimant_history_personal_details_path = os.path.join(self.input_file_path, self.input_dict["Claimant_History_Personal_Details"])
        self.claimant_history_personal_details = self._read_csv_folder(claimant_history_personal_details_path, low_memory=False, converters=self.cols_dtype_dict)
        claimant_history_skill_details_path = os.path.join(self.input_file_path, self.input_dict["Claimant_History_Skill_Details"])
        self.claimant_history_skill_details = self._read_csv_folder(claimant_history_skill_details_path, low_memory=False, converters=self.cols_dtype_dict)

    def _read_csv_folder(self, input_file_path, glob_pattern="*.csv", **kwrags):
        logger.info(f"Reading csv from folder {input_file_path}")
        df_list = []
        if self.fs.isdir(input_file_path):
            if len(self.fs.ls(input_file_path))>0:
                for fl in self.fs.glob(os.path.join(input_file_path, glob_pattern)):
                    if self.fs.du(fl)>0:
                        with self.fs.open(fl) as f:
                            df = pd.read_csv(f, **kwrags)
                            df_list.append(df)
            else:
                logger.error(f"No file found in {input_file_path}.")
                raise FileNotFoundError(f"No file found in {input_file_path}.")
        else:
            logger.warning(f"{input_file_path} is not a directory.")
            raise ValueError(f"{input_file_path} is not a directory.")
        if df_list:
            return pd.concat(df_list, sort=True).reset_index(drop=True)
        else:
            return pd.DataFrame()

    def pi_missing_treatment(self):
        '''
        Special treatment for bank account and routing number
        It looks like 04a4c7210f9df2d6a80433450b94d9007c97 value represents the missing value for bank account number and 
        routing number and state id. This method replaces these values with np.NaN
        '''
        logger.info("pi_missing_treatment")
        missing_val_map = {'04a4c7210f9df2d6a80433450b94d9007c97':np.NaN}
        replace_dict = {'CLAIMANT_STATE_ID_NO':missing_val_map, 
                   'CLAIMANT_BANK_ROUTING_NUMBER':missing_val_map,
                   'CLAIMANT_BANK_ACCOUNT_NUMBER':missing_val_map}
        self.raw_per_df = self.raw_per_df.replace(to_replace=replace_dict)
        
        # Remove duplicate and ambigious records, drop unwanted columns and rename column
        # Remove row where claimant_id_x and claimant_id_y are not matching. This is already handled in ETL and hence commented in model code.
        # Remove unwanted columns
        cols_to_drop = ['CLAIMANT_MAILING_ADDRESS_LINE_2','CLAIMANT_RESIDENTIAL_ADDRESS_LINE_2',                'CLAIMANT_RESIDENTIAL_COUNTRY','CLAIMANT_MAILING_ADDRESS_COUNTRY','CLAIMANT_EMAIL'
                ]		
        self.raw_per_df = self.raw_per_df.drop(cols_to_drop, axis=1)
        # Rename columns
        self.raw_per_df.rename(columns={"CLAIMANT_ID_x": "CLAIMANT_ID"}, inplace=True)
        # Drop duplicates records
        self.raw_per_df = self.raw_per_df.drop_duplicates(subset = ["CLAIMANT_ID", "CLAIM_APPLICATION_ID"])
    
    def pi_match_feature(self):
        '''
        Bank account, State ID, Address matching, work and residential ZIP match
        '''
        logger.info("pi_match_feature")
        self.raw_per_df['CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID'] = np.floor(pd.to_numeric(self.raw_per_df['CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID'], errors='coerce'))
        self.raw_per_df['CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID'] = self.raw_per_df['CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID'].astype('object')

        self.raw_per_df["CLAIMANT_STATE_ID_EXPIRATION_DATE"] = pd.to_datetime(self.raw_per_df["CLAIMANT_STATE_ID_EXPIRATION_DATE"], format="%Y-%m-%d", errors='coerce').dt.date
        self.raw_per_df["CLAIM_APPLICATION_FILE_DATE"] = pd.to_datetime(self.raw_per_df["CLAIM_APPLICATION_FILE_DATE"], format="%d-%m-%Y").dt.date
        self.raw_per_df["CLAIMANT_DATE_OF_BIRTH"] = pd.to_datetime(self.raw_per_df["CLAIMANT_DATE_OF_BIRTH"], format="%d-%m-%Y").dt.date
        #State Id related features
        self.raw_per_df['IS_ID_MISSING'] = (~self.raw_per_df["CLAIMANT_STATE_ID_ISSUING_STATE"].notnull()).astype('int')
        # The condition checks for notnull based on the concept NaN != NaN  in pandas.
        # REFER https://stackoverflow.com/a/46021212/5462372 for condition in query written below. 
        self.raw_per_df['IS_ID_NOT_LOCAL'] = self.raw_per_df.query("CLAIMANT_STATE_ID_ISSUING_STATE == CLAIMANT_STATE_ID_ISSUING_STATE")["CLAIMANT_STATE_ID_ISSUING_STATE"].ne('MS').astype('int')
        self.raw_per_df['IS_ID_STATE_NOT_MATCHING_RES'] = self.raw_per_df.query("CLAIMANT_STATE_ID_ISSUING_STATE == CLAIMANT_STATE_ID_ISSUING_STATE").apply(lambda x: 0 if x.CLAIMANT_STATE_ID_ISSUING_STATE==x.CLAIMANT_RESIDENTIAL_ADDRESS_STATE else 1, axis=1)                                   
        self.raw_per_df['IS_ID_STATE_NOT_MATCHING_MAIL'] = self.raw_per_df.query("CLAIMANT_STATE_ID_ISSUING_STATE == CLAIMANT_STATE_ID_ISSUING_STATE").apply(lambda x: 0 if x.CLAIMANT_STATE_ID_ISSUING_STATE==x.CLAIMANT_MAILING_ADDRESS_STATE else 1, axis=1)
        self.raw_per_df['ID_EXPIRED'] = self.raw_per_df.query("CLAIMANT_STATE_ID_EXPIRATION_DATE == CLAIMANT_STATE_ID_EXPIRATION_DATE").apply(lambda x: is_card_expired(x.CLAIMANT_STATE_ID_EXPIRATION_DATE, x.CLAIM_APPLICATION_FILE_DATE),axis=1)
        
        #Claimant address related features
        self.raw_per_df['IS_MAIL_STATE_NOT_LOCAL'] = self.raw_per_df["CLAIMANT_MAILING_ADDRESS_STATE"].ne('MS').astype('int')
        self.raw_per_df['IS_RES_STATE_NOT_LOCAL'] = self.raw_per_df["CLAIMANT_RESIDENTIAL_ADDRESS_STATE"].ne('MS').astype('int')
        logger.debug(f"raw_per_df shape IS_RES_STATE_NOT_LOCAL {self.raw_per_df.shape}")
        self.raw_per_df['RES_ADDR'] = self.raw_per_df['CLAIMANT_RESIDENTIAL_ADDRESS_LINE_1'].astype(str)+                                     ' ' + self.raw_per_df['CLAIMANT_RESIDENTIAL_ADDRESS_CITY'].astype(str)+                                     ' ' + self.raw_per_df['CLAIMANT_RESIDENTIAL_ADDRESS_STATE'].astype(str)+                                     ' ' + self.raw_per_df['CLAIMANT_RESIDENTIAL_ADDRESS_ZIP'].astype(str)
        self.raw_per_df['MAIL_ADDR'] = self.raw_per_df['CLAIMANT_MAILING_ADDRESS_LINE_1'].astype(str)+                                     ' ' + self.raw_per_df['CLAIMANT_MAILING_ADDRESS_CITY'].astype(str)+                                     ' ' + self.raw_per_df['CLAIMANT_MAILING_ADDRESS_STATE'].astype(str)+                                     ' ' + self.raw_per_df['CLAIMANT_MAILING_ADDRESS_ZIP'].astype(str)

        temp_df = self.raw_per_df[['CLAIMANT_ID','CLAIM_APPLICATION_ID','RES_ADDR']].drop_duplicates(['CLAIMANT_ID','RES_ADDR'],keep= 'last')
        dict_res_address = dict(temp_df.RES_ADDR.value_counts() > 1)
        self.raw_per_df['LIVING_TOGETHER'] = self.raw_per_df['RES_ADDR'].map(dict_res_address)

        temp_df = self.raw_per_df[['CLAIMANT_ID','CLAIM_APPLICATION_ID','MAIL_ADDR']].drop_duplicates(['CLAIMANT_ID','MAIL_ADDR'],keep= 'last')
        dict_mail_address = dict(temp_df.MAIL_ADDR.value_counts() > 1)
        self.raw_per_df['CORRESPONDING_TOGETHER'] = self.raw_per_df['MAIL_ADDR'].map(dict_mail_address)
        self.raw_per_df['NOT_SAME_MAIL_RES_ADDR'] = self.raw_per_df.apply(lambda x: match_column(x.RES_ADDR,x.MAIL_ADDR,90), axis=1)
        self.raw_per_df['NOT_SAME_WORK_PRE_RES_ZIP'] = self.raw_per_df.apply(lambda x: match_column(x.CLAIMANT_RESIDENTIAL_ADDRESS_ZIP,x.CLAIMANT_PREFERRED_WORK_ZIP_CODE,100), axis=1)

        #Claimant personal related features
        # Get the age
        self.raw_per_df["CLAIMANT_DATE_OF_BIRTH"] = pd.to_datetime(self.raw_per_df["CLAIMANT_DATE_OF_BIRTH"])
        self.raw_per_df["CLAIM_APPLICATION_FILE_DATE"] = pd.to_datetime(self.raw_per_df["CLAIM_APPLICATION_FILE_DATE"])
        self.raw_per_df["AGE"] = (self.raw_per_df["CLAIM_APPLICATION_FILE_DATE"] - self.raw_per_df["CLAIMANT_DATE_OF_BIRTH"]).astype('<m8[Y]')
        self.raw_per_df['SENIORCITIZEN'] = self.raw_per_df["AGE"].gt(60).astype('int')
        self.raw_per_df['AGE_BUCKET'] = pd.cut(self.raw_per_df['AGE'], [-999, 16, 40, 60, 80, 999], labels=['< 16', '16-40', '41-60','61-80','> 80'])
        self.raw_per_df["INVALID_DOB"] = (self.raw_per_df["AGE"].lt(16) | self.raw_per_df["AGE"].gt(80)).astype('int')
        #Claimant Bank related features
        # This condition in query checks the column isnull using the condition in query. REFER https://stackoverflow.com/a/46021212/5462372 for below code in query. 
        self.raw_per_df['IS_ACC_MISSING'] = self.raw_per_df.query("CLAIMANT_BANK_ACCOUNT_NUMBER != CLAIMANT_BANK_ACCOUNT_NUMBER")["CLAIMANT_PAYMENT_MODE"].eq("D").astype('int')
        self.raw_per_df['IS_ACC_MISSING'].fillna(0, inplace=True)
        #Check if there is any attempt to bank fruad using payment mode of Debit card and mailing & residential address is not matching
        self.raw_per_df['IS_BANKING_FRAUD'] = (self.raw_per_df["CLAIMANT_PAYMENT_MODE"].eq("D") & self.raw_per_df["NOT_SAME_MAIL_RES_ADDR"].eq(0)).astype('int')
        #Required for visualization purpose
        self.raw_per_df["YEAR"] = self.raw_per_df["CLAIM_APPLICATION_FILE_DATE"].dt.year
        self.raw_per_df["MONTH"] = self.raw_per_df["CLAIM_APPLICATION_FILE_DATE"].dt.month
        self.raw_per_df["WEEK"] = self.raw_per_df["CLAIM_APPLICATION_FILE_DATE"].dt.weekofyear
        self.raw_per_df['YEAR_MONTH'] = self.raw_per_df['CLAIM_APPLICATION_FILE_DATE'].dt.strftime('%Y-%m')
        self.raw_per_df["YEAR_WEEK"] = self.raw_per_df['CLAIM_APPLICATION_FILE_DATE'].dt.strftime('%Y-%W')

    def pi_theft_feature(self):
        '''
        Email, Phone, bank account  and state id theft
        '''
        logger.info("pi_theft_feature")
        temp_df = self.raw_per_df[['CLAIMANT_ID','CLAIM_APPLICATION_ID','CLAIMANT_PHONE_NUMBER']].drop_duplicates(['CLAIMANT_ID','CLAIMANT_PHONE_NUMBER'],keep= 'last')
        dict_phone = dict(temp_df.CLAIMANT_PHONE_NUMBER.value_counts() > 3)
        self.raw_per_df['PHONE_THEFT'] = self.raw_per_df['CLAIMANT_PHONE_NUMBER'].map(dict_phone)

        temp_df = self.raw_per_df[['CLAIMANT_ID','CLAIM_APPLICATION_ID','EMAIL']].query('EMAIL==EMAIL').drop_duplicates(['CLAIMANT_ID','EMAIL'],keep= 'last')
        dict_email = dict(temp_df.EMAIL.value_counts() > 3)
        self.raw_per_df['EMAIL_THEFT'] = self.raw_per_df['EMAIL'].map(dict_email)

        temp_df = self.raw_per_df[['CLAIMANT_ID','CLAIM_APPLICATION_ID','CLAIMANT_BANK_ACCOUNT_NUMBER']].query('CLAIMANT_BANK_ACCOUNT_NUMBER==CLAIMANT_BANK_ACCOUNT_NUMBER').drop_duplicates(['CLAIMANT_ID','CLAIMANT_BANK_ACCOUNT_NUMBER'],keep= 'last')
        dict_acc = dict(temp_df.CLAIMANT_BANK_ACCOUNT_NUMBER.value_counts()>1)
        self.raw_per_df['ACC_NUMBER_THEFT'] = self.raw_per_df['CLAIMANT_BANK_ACCOUNT_NUMBER'].map(dict_acc)

        temp_df = self.raw_per_df[['CLAIMANT_ID','CLAIM_APPLICATION_ID','CLAIMANT_STATE_ID_NO']].query('CLAIMANT_STATE_ID_NO==CLAIMANT_STATE_ID_NO').drop_duplicates(['CLAIMANT_ID','CLAIMANT_STATE_ID_NO'],keep= 'last')
        dict_stateid = dict(temp_df.CLAIMANT_STATE_ID_NO.value_counts()>1)
        self.raw_per_df['STATE_ID_THEFT'] = self.raw_per_df['CLAIMANT_STATE_ID_NO'].map(dict_stateid)

        del temp_df
        gc.collect()
        
    def freq_addr_change(self):
        '''
        Frequent Mailing and residential address, bank account and state id change
        '''
        logger.info("freq_addr_change")
        per_resaddr_cnt_df = self.raw_per_df.groupby('CLAIMANT_ID')['RES_ADDR'].nunique().reset_index().rename(columns={'RES_ADDR':'RES_ADDR_COUNT'})
        per_resaddr_cnt_df['IS_FREQ_RES_ADDR_CHANGE'] = 0
        per_resaddr_cnt_df['IS_FREQ_RES_ADDR_CHANGE'] = per_resaddr_cnt_df["RES_ADDR_COUNT"].gt(2).astype('int')

        self.raw_per_df = self.raw_per_df.merge(per_resaddr_cnt_df[['CLAIMANT_ID','IS_FREQ_RES_ADDR_CHANGE']], on='CLAIMANT_ID', how='left')
        del per_resaddr_cnt_df

        per_mailaddr_cnt_df = self.raw_per_df.groupby('CLAIMANT_ID')['MAIL_ADDR'].nunique().reset_index().rename(columns={'MAIL_ADDR':'MAIL_ADDR_COUNT'})
        per_mailaddr_cnt_df['IS_FREQ_MAIL_ADDR_CHANGE'] = 0
        per_mailaddr_cnt_df['IS_FREQ_MAIL_ADDR_CHANGE'] = per_mailaddr_cnt_df["MAIL_ADDR_COUNT"].gt(2).astype('int')

        self.raw_per_df = self.raw_per_df.merge(per_mailaddr_cnt_df[['CLAIMANT_ID','IS_FREQ_MAIL_ADDR_CHANGE']], on='CLAIMANT_ID', how='left')
        del per_mailaddr_cnt_df

       
        per_bankacc_cnt_df = self.raw_per_df.groupby('CLAIMANT_ID')['CLAIMANT_BANK_ACCOUNT_NUMBER'].nunique().reset_index().rename(columns={'CLAIMANT_BANK_ACCOUNT_NUMBER':'BANK_ACC_COUNT'})
        per_bankacc_cnt_df['IS_FREQ_BANK_ACC_CHANGE'] = 0
        per_bankacc_cnt_df['IS_FREQ_BANK_ACC_CHANGE'] = per_bankacc_cnt_df["BANK_ACC_COUNT"].gt(1).astype('int')

        self.raw_per_df = self.raw_per_df.merge(per_bankacc_cnt_df[['CLAIMANT_ID','IS_FREQ_BANK_ACC_CHANGE']], on='CLAIMANT_ID', how='left')
        del per_bankacc_cnt_df

        per_stateid_cnt_df = self.raw_per_df.groupby('CLAIMANT_ID')['CLAIMANT_STATE_ID_NO'].nunique().reset_index().rename(columns={'CLAIMANT_STATE_ID_NO':'STATE_ID_COUNT'})
        per_stateid_cnt_df['IS_FREQ_STATE_ID_CHANGE'] = 0
        per_stateid_cnt_df['IS_FREQ_STATE_ID_CHANGE'] = per_stateid_cnt_df["STATE_ID_COUNT"].gt(1).astype('int')

        self.raw_per_df = self.raw_per_df.merge(per_stateid_cnt_df[['CLAIMANT_ID','IS_FREQ_STATE_ID_CHANGE']], on='CLAIMANT_ID', how='left')
        del per_stateid_cnt_df
        
    def pi_suspicious(self):
        '''
        Features for suspicious DOB, skill set and education level from historical activities
        '''
        logger.info("pi_suspicious")
        
        claimant_df = self.raw_per_df.copy()
        per_dob_cnt_df = self.claimant_history_personal_details.groupby('CLAIMANT_ID')['CLAIMANT_DATE_OF_BIRTH'].nunique().reset_index().rename(columns={'CLAIMANT_DATE_OF_BIRTH':'DOB_COUNT'})
        per_dob_cnt_df['IS_suspicious_DOB'] = 0
        per_dob_cnt_df['IS_suspicious_DOB'] = per_dob_cnt_df["DOB_COUNT"].gt(1).astype('int')

        claimant_df = claimant_df.merge(per_dob_cnt_df, on='CLAIMANT_ID', how='left')
        del per_dob_cnt_df
        
        #education change change - If the claimant shows less qualified compared to last application ID
        self.claimant_history_personal_details.dropna(axis=0, subset=["CLAIMANT_EDUCATION_LEVEL_CODE"], inplace=True)
        self.claimant_history_personal_details.sort_values(by=['CLAIMANT_ID','CLAIM_APPLICATION_ID'], inplace=True)

        #educational level more than what they have mentioned last time

        self.claimant_history_personal_details["CLAIMANT_EDUCATION_LEVEL_CODE_typed"] = self.claimant_history_personal_details["CLAIMANT_EDUCATION_LEVEL_CODE"].replace({"BD": "15", "HD": "15", "AD": "15", "GD": "15",
                                                                                   "PD": "20","PD  ": "20", "CC": "12","CC  ": "12", "MD": "17","MD  ": "17", "HD  ": "15", "BD  ": "15","AD  ": "15", "GD  ": "15"})
        self.claimant_history_personal_details["CLAIMANT_EDUCATION_LEVEL_CODE_typed"] = self.claimant_history_personal_details["CLAIMANT_EDUCATION_LEVEL_CODE_typed"].astype("int")

        #education change change
        # Get non-duplicate rows based on CLAIMANT_ID column value and from that create a dict 
        # with CLAIMANT_ID as key and CLAIMANT_EDUCATION_LEVEL_CODE_typed as value. This dict represents
        # the first education level registered for each CLAIMANT_ID.
        edu_level_dict = self.claimant_history_personal_details.loc[~self.claimant_history_personal_details["CLAIMANT_ID"].duplicated(), ["CLAIMANT_ID", "CLAIMANT_EDUCATION_LEVEL_CODE_typed"]].set_index("CLAIMANT_ID")["CLAIMANT_EDUCATION_LEVEL_CODE_typed"].to_dict()
        # Create a temporary dataframe with only duplicate rows based on CLAIMANT_ID column value.
        # This represents the subsequent education levels registered for each CLAIMANT_ID.
        duplicates = self.claimant_history_personal_details.loc[self.claimant_history_personal_details["CLAIMANT_ID"].duplicated(), ["CLAIMANT_ID", "CLAIMANT_EDUCATION_LEVEL_CODE_typed"]]
        # Map a new column PREV_EDUCATION_LEVEL using CLAIMANT_ID and edu_level_dict.
        duplicates["PREV_EDUCATION_LEVEL"] = duplicates["CLAIMANT_ID"].map(edu_level_dict)
        # Select CLAIMANT_ID into a list where PREV_EDUCATION_LEVEL is greater than current education level (CLAIMANT_EDUCATION_LEVEL_CODE_typed)
        fraud_claimant_id_edu_level = duplicates.loc[duplicates["PREV_EDUCATION_LEVEL"] > duplicates["CLAIMANT_EDUCATION_LEVEL_CODE_typed"], "CLAIMANT_ID"].drop_duplicates().tolist()
        # fraud_claimant_id_edu_level         

        # Claimant_History_Personal_Details_EDU

        claimant_df["Is_suspicious_education"] = 0
        claimant_df.loc[(claimant_df["CLAIMANT_ID"].isin(fraud_claimant_id_edu_level)), "Is_suspicious_education"] = 1
        
        #ambiguous skill details - If the claimants current application skill details are not matching with their previous mentioned skill details
        # missing_value_describe(Claimant_History_Skill_Details)

        claim_application_skill_dict = self.claimant_history_skill_details.groupby("CLAIM_APPLICATION_ID")["SKILL_SET_CODE"].apply(list).to_dict()


        # claim_application_skill_dict
        # concatenate skill code groupby CLAIM_APPLICATION_ID
        # now for each claimant id, compare the skills from the previous claimant if any matches, add the skill code against that claimant id
        # if it doesn't matches then add that claimant id to the list


        claimant_skill_dict = {}
        fraud_claimant_id_skill_set = []
        for index, value in self.claimant_history_skill_details.iterrows():
            claim_application_id = value["CLAIM_APPLICATION_ID"]
            claimant_id = value["CLAIMANT_ID"]
            claimant_skills = claim_application_skill_dict[claim_application_id]
            if claimant_id in claimant_skill_dict.keys():
                prev_claimant_skills = claimant_skill_dict[claimant_id]
                if (set(claimant_skills) & set(prev_claimant_skills)):
                    claimant_skill_dict[claimant_id] = claimant_skills + prev_claimant_skills
                elif (len(prev_claimant_skills) == 1):
                    claimant_skill_dict[claimant_id] = claimant_skills + prev_claimant_skills
                else:    
                    fraud_claimant_id_skill_set.append(claimant_id)
            else:
                claimant_skill_dict[claimant_id] = claimant_skills
        # fraud_claimant_id_skill_set    

        # len(set(fraud_claimant_id_skill_set))

        claimant_df["Is_suspicious_skill_history"] = 0
        claimant_df.loc[(claimant_df["CLAIMANT_ID"].isin(fraud_claimant_id_skill_set)), "Is_suspicious_skill_history"] = 1
        self.raw_per_df = claimant_df
        del claimant_df
        gc.collect()
    
    def skill_features(self):
        '''
        Skill related feature engineering
        '''
        logger.info("skill_features")
        raw_per_df = self.raw_per_df
        skill_df = self.skill_df
        #Need to get unique ones to make operations only for claimants that are present in claimant details file
        skill_df.dropna(axis=0,inplace=True)

        #Need to get unique ones to make operations only for claimants that are present in claimant details file
        unique_claimant_id = list(raw_per_df["CLAIMANT_ID"])

        #Drop dupliclate data with same claimant id and CLAIMANT_SKILL_CODE
        skill_df.drop_duplicates(subset=["CLAIMANT_ID", "CLAIMANT_SKILL_CODE"], inplace=True)
        #Remove records where claimant id not present in personal details data
        skill_df = skill_df.loc[(skill_df["CLAIMANT_ID"].isin(unique_claimant_id))]

        skill_df['IS_SKILL_PREFERENCE_NOT_MATCHING'] = skill_df["IS_PRIMARY_SKILL_DECLARED_BY_CLAIMANT"].eq(skill_df["IS_PREFERRED_SKILL_DECLARED_BY_CLAIMANT"]).astype('int')

        #find sum of the work experience for each claimant
        pref_work_exp_df = skill_df.query('IS_PREFERRED_SKILL_DECLARED_BY_CLAIMANT==1').groupby(["CLAIMANT_ID"])['WORK_EXPERIENCE_IN_MONTHS'].agg("sum").reset_index().rename(columns={'WORK_EXPERIENCE_IN_MONTHS':'PREF_WORK_EXP'})
        skill_df = skill_df.merge(pref_work_exp_df, how='left', on='CLAIMANT_ID')

        skill_df['SKILL_COUNT'] = skill_df.groupby("CLAIMANT_ID")['CLAIMANT_SKILL_CODE'].transform("count")

        high_skill_count = max(skill_df.SKILL_COUNT.unique())

        skill_df['VERY_HIGH_SKILL_COUNT'] = skill_df["SKILL_COUNT"].ge(high_skill_count).astype('int')

        #Removing records with no values in CLAIAMANT_SKILL_DESCRIPTION and CLAIMANT_SKILL_CODE
        claimant_primary_skill = skill_df.loc[(skill_df["IS_PRIMARY_SKILL_DECLARED_BY_CLAIMANT"] == 1.0)]
        claimant_primary_skill = claimant_primary_skill.sort_values("CLAIM_APPLICATION_ID").groupby("CLAIMANT_ID").tail(1)

        #Cumulative work experience
        cumulative_work_exp = skill_df.groupby(['CLAIMANT_ID','CLAIMANT_SKILL_CODE'])['WORK_EXPERIENCE_IN_MONTHS'].max().reset_index(level=0)
        cumulative_work_exp = cumulative_work_exp.groupby(['CLAIMANT_ID'])['WORK_EXPERIENCE_IN_MONTHS'].sum().reset_index()

        #cumulative_work_exp

        claimant_primary_skill = pd.merge(claimant_primary_skill, cumulative_work_exp, how="inner", on="CLAIMANT_ID")
        #claimant_primary_skill
        del cumulative_work_exp
        claimant_primary_skill.drop(["WORK_EXPERIENCE_IN_MONTHS_x", "IS_PRIMARY_SKILL_DECLARED_BY_CLAIMANT", "IS_PREFERRED_SKILL_DECLARED_BY_CLAIMANT"], axis=1, inplace=True)
        claimant_primary_skill.rename(columns={"WORK_EXPERIENCE_IN_MONTHS_y": "WORK_EXPERIENCE_IN_MONTHS"}, inplace=True)
        
        self.claimant_skill_df = claimant_primary_skill
        gc.collect()
    
    def employer_features(self):
        '''
        Employer data preparation
        '''
        logger.info("employer_features")
        missing_val_map = {'04a4c7210f9df2d6a80433450b94d9007c97':np.NaN}
        replace_dict = {'EMPLOYER_NAME':missing_val_map}
        self.employer_df = self.employer_df.replace(to_replace=replace_dict)
        employer_cols_to_keep = ["EMPLOYER_ID", "REGISTRATION_DATE","EMPLOYER_INDUSTRY_CODE","EMPLOYER_INDUSTRY_DESC",                         "EMPLOYER_NAME","MAIL_ADDRESS_STATE","MAIL_ADDRESS_COUNTRY"]
        self.employer_df = self.employer_df[employer_cols_to_keep]
        self.employer_df["REGISTRATION_DATE"].fillna("2020-09-01", inplace=True)
        self.employer_df["EMPLOYER_INDUSTRY_CODE"].fillna(-1, inplace=True)
        self.employer_df["EMPLOYER_INDUSTRY_DESC"].fillna("NA", inplace=True)
        self.employer_df["EMPLOYER_AGE"] = self.employer_df["REGISTRATION_DATE"].apply(lambda x : calculate_age(datetime.strptime(x, "%Y-%m-%d")))
        self.employer_df["EMPLOYER_AGE"] = self.employer_df["EMPLOYER_AGE"].replace([2020], -1)
        self.employer_df['EMP_AGE_BUCKET'] = pd.cut(self.employer_df['EMPLOYER_AGE'], [-999, 1, 5, 10, 15, 100], labels=['0-1', '1-5', '5-10','10-15','15-100'])
    
    def claimant_prev_fraud_history(self, temp_df):
        logger.info("claimant_prev_fraud_history")
        is_train = self.is_train
        #load the raw dataset if 
        is_prv_fraud_commit = []
        previous_fraud_commit_claimants = {}
        # logger.info (is_train)
        if is_train == True:
            temp_df = temp_df[['CLAIMANT_ID','CLAIM_APPLICATION_ID','FRAUD','CLAIM_APPLICATION_FILE_DATE']]
            temp_df["is_prv_fraud_commit"] = temp_df.groupby("CLAIMANT_ID")['FRAUD'].shift().where(lambda s: s.eq(1))
            is_prv_fraud_commit = temp_df.groupby("CLAIMANT_ID")["is_prv_fraud_commit"].ffill().fillna(0).astype("int").tolist()
        else:
            raw_per_df = self.history_df
            raw_per_df = raw_per_df.loc[raw_per_df["CLAIM_APPLICATION_FILE_DATE"] >= self.input_dict["history_data_start_date"]]
            previous_fraud_commit_claimants = {x:1 for x in raw_per_df.loc[raw_per_df["FRAUD"]==1, "CLAIMANT_ID_x"].to_numpy()}
            is_prv_fraud_commit = temp_df["CLAIMANT_ID"].map(previous_fraud_commit_claimants).fillna(0).tolist()
        logger.info(f"Length of previous_fraud_commit_claimants {len(previous_fraud_commit_claimants)}")
        return is_prv_fraud_commit
                
    def suspicious_employer(self, claimant_per_emp_sk_df):
        logger.info("suspicious_employer")
        is_train = self.is_train
        #in case of prediction it will not have the fraud labels, hence using the raw data created separately
        if is_train == False:
            claimant_temp_df = claimant_per_emp_sk_df.copy()
            claimant_per_emp_sk_df = self.history_df
            claimant_per_emp_sk_df = claimant_per_emp_sk_df.loc[claimant_per_emp_sk_df["CLAIM_APPLICATION_FILE_DATE"] >= self.input_dict["history_data_start_date"]]
            
        #TOCHECK changed to loc from query
        claim_cnt = claimant_per_emp_sk_df.groupby(['EMPLOYER_ID'])['CLAIM_APPLICATION_ID'].count().reset_index().rename(columns={'CLAIM_APPLICATION_ID':'CLAIM_CNT'})
        fraud_cnt = claimant_per_emp_sk_df.loc[claimant_per_emp_sk_df["FRAUD"] == 1].groupby(['EMPLOYER_ID'])['CLAIM_APPLICATION_ID'].count().reset_index().rename(columns={'CLAIM_APPLICATION_ID':'FRAUD_CNT'})
        claim_fraud_cnt = pd.merge(claim_cnt, fraud_cnt, how="left", on=["EMPLOYER_ID"])

        claim_fraud_cnt.fillna(0, inplace=True)
        claim_fraud_cnt['FRAUD_RATIO'] = round(claim_fraud_cnt.FRAUD_CNT/claim_fraud_cnt.CLAIM_CNT, 2)
        claim_fraud_cnt['IS_SUSPICIOUS_EMPLOYER'] = claim_fraud_cnt["FRAUD_RATIO"].gt(0.5).astype('int')
        
        if is_train == False:
            claimant_per_emp_sk_df = claimant_temp_df.copy()
            del claimant_temp_df
        del claim_cnt, fraud_cnt
        
        claimant_per_emp_sk_df = pd.merge(claimant_per_emp_sk_df, claim_fraud_cnt[['EMPLOYER_ID','IS_SUSPICIOUS_EMPLOYER']], how="left", on=["EMPLOYER_ID"])
        claimant_per_emp_sk_df['IS_SUSPICIOUS_EMPLOYER'].fillna(0,inplace=True)
        logger.info ("claimant_per_emp_sk_df suspicious_employer function")
        return claimant_per_emp_sk_df
        
    
    def employment_features(self):
        logger.info("employment_features")
        is_train = self.is_train
        '''
        Employment data preparation      
        '''
        employment_df = self.employment_df
        employer_df = self.employer_df
        raw_per_df = self.raw_per_df
        claimant_skill_df = self.claimant_skill_df
        missing_val_map = {'04a4c7210f9df2d6a80433450b94d9007c97':np.NaN}
        replace_dict = {'CLAIM_APPLICATION_EMPLOYER_NAME':missing_val_map}
        employment_df = employment_df.replace(to_replace=replace_dict)
        
        #Remove columns which are not making any sense  
        emp_cols_to_drop = ['EMPLOYER_REGISTERD_NAME_DIFFERS','CLAIM_APPLICATION_EMPLOYER_ADDRESS_LINE2',                    'CLAIMANT_WORKED_IN_CITY','CLAIMANT_WORKED_IN_STATE',                    'CLAIM_APPLICATION_EMPLOYER_ADDRESS_LINE1','CLAIM_APPLICATION_EMPLOYER_ZIP',                    'CLAIM_APPLICATION_EMPLOYER_CITY']
        employment_df = clean_data(employment_df,emp_cols_to_drop)
        #Imputing missing values
        employment_df.drop_duplicates(subset=["CLAIMANT_ID", "CLAIM_APPLICATION_ID"], inplace=True)        
        employment_df["CLAIMANT_MENTIONED_PAY_RATE"].fillna(0, inplace=True)
        # Impute the missing pay rate by hour as the average PAY RATE is ~7
        employment_df["CLAIMANT_MENTIONED_PAY_RATE_FREQUENCY"].fillna("MONT", inplace=True)
        # The below feature is matching for all now.
        employment_df['IS_EMPLOYER_MISSING'] = (~(employment_df["EMPLOYER_ID"].notnull() & employment_df["CLAIM_APPLICATION_EMPLOYER_NAME"].notnull())).astype('int')
        #Merge employer and employment details for further calcualtion
        employer_employee_df = pd.merge(employment_df, employer_df, how="left", on=["EMPLOYER_ID"])
        
        del employment_df
        del employer_df
        gc.collect()
        employer_employee_df["CLAIMANT_MENTIONED_PAY_RATE_FREQUENCY"] = employer_employee_df["CLAIMANT_MENTIONED_PAY_RATE_FREQUENCY"].map({" ": "", "HOUR": 720.0, "WEEK" :4.0, 
                                                                                    "YEAR" : 0.083, "BIWK": 2.0, "MONT":1.0, "DAY":30.0, "DAY ":30.0}).astype("float")
        # All mismatched values will become 1.0 (MONT) by default.
        employer_employee_df["CLAIMANT_MENTIONED_PAY_RATE_FREQUENCY"].fillna(1.0, inplace=True)
        employer_employee_df["CLAIMANT_MENTIONED_PAY"] = employer_employee_df.CLAIMANT_MENTIONED_PAY_RATE * employer_employee_df.CLAIMANT_MENTIONED_PAY_RATE_FREQUENCY
 
        emp_cols_to_keep = ['CLAIMANT_ID', 'CLAIM_APPLICATION_ID','CLAIMANT_MENTIONED_PAY','IS_EMPLOYER_MISSING',                    'CLAIM_APPLICATION_EMPLOYER_STATE','CLAIM_APPLICATION_EMPLOYER_ADDRESS_CITY',                    'SEPARATION_REASON_WITH_EMPLOYER','JOBTITLE_MENTIONED_BY_EMPLOYER',"EMPLOYER_ID",'EMP_AGE_BUCKET',                    'EMPLOYER_INDUSTRY_CODE',"EMPLOYER_INDUSTRY_DESC",'EMPLOYER_WAS_NOT_PRESENT_IN_SYSTEM','EMPLOYER_TYPE','EMPLOYER_NAME']

        claimant_emp = employer_employee_df[emp_cols_to_keep]
        
        '''
        Merge Employment and personal details
        '''
        claimant_per_emp_df = pd.merge(raw_per_df, claimant_emp, how="left", on=["CLAIMANT_ID", "CLAIM_APPLICATION_ID"])
        
        '''
        Merge again with skill dataset using claimant_id column
        '''
        claimant_per_emp_sk_df = pd.merge(claimant_per_emp_df, claimant_skill_df, how="left", on=["CLAIMANT_ID", "CLAIM_APPLICATION_ID"])
        
        claimant_per_emp_sk_df['IS_WORK_EXP_AMBIGIOUS'] = claimant_per_emp_sk_df["WORK_EXPERIENCE_IN_MONTHS"].gt(claimant_per_emp_sk_df["AGE"].multiply(12).subtract(16)).astype('int')
        
        '''
        County wise education wise skill wise earning on basic pay-> suspicious employer
        
        '''

        claimant_per_emp_sk_df["CLAIMANT_EDUCATION_LEVEL_CODE"] = claimant_per_emp_sk_df.query('CLAIMANT_EDUCATION_LEVEL_CODE == CLAIMANT_EDUCATION_LEVEL_CODE')["CLAIMANT_EDUCATION_LEVEL_CODE"].replace({"BD": "15", "HD": "15", "AD": "15", "GD": "15",
                                                                                   "PD": "20","PD  ": "20", "CC": "12","CC  ": "12", "MD": "17","MD  ": "17", "HD  ": "15", "BD  ": "15","AD  ": "15", "GD  ": "15"})
        claimant_per_emp_sk_df["CLAIMANT_EDUCATION_LEVEL_CODE"] = claimant_per_emp_sk_df.query('CLAIMANT_EDUCATION_LEVEL_CODE == CLAIMANT_EDUCATION_LEVEL_CODE')["CLAIMANT_EDUCATION_LEVEL_CODE"].astype("int")
        skill_education_pay_df = claimant_per_emp_sk_df.groupby(["CLAIMANT_SKILL_CODE", "CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID", "CLAIMANT_EDUCATION_LEVEL_CODE"]).agg({"CLAIMANT_MENTIONED_PAY": [q1_quantile, q2_quantile, q3_quantile, "max"]})

        skill_education_pay_df.columns = skill_education_pay_df.columns.droplevel(0)
        skill_education_pay_df.reset_index(inplace=True)
        skill_education_pay_df.dropna(axis=0, inplace=True)
        
        skill_education_pay_df["upper_count"] = skill_education_pay_df["q3_quantile"].add(skill_education_pay_df["q3_quantile"].subtract(skill_education_pay_df["q1_quantile"]).multiply(1.5))
        skill_education_pay_df.drop_duplicates(subset=["CLAIMANT_SKILL_CODE", "CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID", "CLAIMANT_EDUCATION_LEVEL_CODE"], inplace=True)
        upper_count_skill_education_dict = {}

        skill_education_pay_df["county_skill_education"] = skill_education_pay_df["CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID"].astype(str)+                 ":" + skill_education_pay_df["CLAIMANT_SKILL_CODE"].astype(str) +                 ":" + skill_education_pay_df["CLAIMANT_EDUCATION_LEVEL_CODE"].astype(float).astype(str)
        upper_count_skill_education_dict = skill_education_pay_df.set_index("county_skill_education")["q3_quantile"].to_dict()
        # upper_count_skill_education_dict

        claimant_per_emp_sk_df["county_skill_education"] = claimant_per_emp_sk_df["CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID"].astype(str)+                 ":" + claimant_per_emp_sk_df["CLAIMANT_SKILL_CODE"].astype(str) +                 ":" + claimant_per_emp_sk_df["CLAIMANT_EDUCATION_LEVEL_CODE"].astype(float).astype(str)
            
        m = claimant_per_emp_sk_df["county_skill_education"].map(upper_count_skill_education_dict)
        m[pd.isnull(m)]=0
        anomaly_arr = np.where(m,             claimant_per_emp_sk_df["CLAIMANT_MENTIONED_PAY"].gt(claimant_per_emp_sk_df["county_skill_education"].map(upper_count_skill_education_dict)).astype('int'),             -1)
        claimant_per_emp_sk_df["is_suspicious_wage_base_period"] = anomaly_arr

        # unemployment_df  
        '''
        Get the fraud history
        
        '''
        claimant_per_emp_sk_df = claimant_per_emp_sk_df.sort_values(by=['CLAIMANT_ID','CLAIM_APPLICATION_FILE_DATE'])
        #few features require the fraud and employer id data
        if not is_train:
            input_dict = self.input_dict
            logger.info("Reading files historical claimant personal details.")
            raw_history_df_path = os.path.join(self.input_file_path, input_dict["raw_per_df_history"])
            raw_history_df = self._read_csv_folder(raw_history_df_path, low_memory=False, usecols = ['CLAIMANT_ID_x','CLAIM_APPLICATION_FILE_DATE','CLAIM_APPLICATION_ID', 'FRAUD'], converters=self.cols_dtype_dict)
            if not self.interim_df.empty:
                logger.info("Merging claimnat personal details interim data during scoring")
                interim_df = self.interim_df[['CLAIMANT_ID_x','CLAIM_APPLICATION_FILE_DATE','CLAIM_APPLICATION_ID', 'FRAUD']]
                raw_history_df = pd.concat([raw_history_df, interim_df], sort=True).reset_index(drop=True)
            raw_emp_history_df = self.employment_df[['CLAIMANT_ID','EMPLOYER_ID']]
            raw_history_df = pd.merge(raw_history_df, raw_emp_history_df, how="left", left_on="CLAIMANT_ID_x", right_on="CLAIMANT_ID")
            raw_history_df["EMPLOYER_ID"].fillna(0, inplace=True)
            raw_history_df["EMPLOYER_ID"] = raw_history_df["EMPLOYER_ID"].astype(int)
            raw_history_df.drop("CLAIMANT_ID", axis=1, inplace=True) #redundant CLAIMANT_ID columns exist
            raw_history_df["CLAIM_APPLICATION_FILE_DATE"] = pd.to_datetime(raw_history_df["CLAIM_APPLICATION_FILE_DATE"], format="%d-%m-%Y")
            self.history_df = raw_history_df
            del raw_emp_history_df, raw_history_df
        
        is_prv_fraud_commit = self.claimant_prev_fraud_history(claimant_per_emp_sk_df)
        claimant_per_emp_sk_df['HAS_FRAUD_HISTORY'] = is_prv_fraud_commit
        
        '''
        Check Fraud Employer
        '''
        claimant_per_emp_sk_df = self.suspicious_employer(claimant_per_emp_sk_df)
        return claimant_per_emp_sk_df
    
    
    def data_preparation(self): 
        '''
        Special treatment for bank account and routing number
        It looks like 04a4c7210f9df2d6a80433450b94d9007c97 value represents the missing value for bank account number and 
        routing number and state id
        Remove duplicate and ambigious records , drop unwanted columns and rename column
        '''
        logger.info("data_preparation")
        logger.debug(f"raw_per_df {self.raw_per_df.shape} Skill df {self.skill_df.shape}\n             employment_df {self.employer_df.shape}  employer_df {self.employer_df.shape}\n             Claimant_History_Personal_Details {self.claimant_history_personal_details.shape}\n             Claimant_History_Skill_Details {self.claimant_history_skill_details.shape}")
        self.pi_missing_treatment()
        
        '''
        Feature engineering for Personal Data
        Bank account, State ID, Address matching, work and residential ZIP match
        ''' 
        logger.debug(f"raw_per_df {self.raw_per_df.shape} Skill df {self.skill_df.shape}\n             employment_df {self.employer_df.shape}  employer_df {self.employer_df.shape}\n             Claimant_History_Personal_Details {self.claimant_history_personal_details.shape}\n             Claimant_History_Skill_Details {self.claimant_history_skill_details.shape}")
        self.pi_match_feature()

        '''
        Email, Phone, bank account  and state id theft
        ''' 
        logger.debug(f"raw_per_df {self.raw_per_df.shape} Skill df {self.skill_df.shape}\n             employment_df {self.employer_df.shape}  employer_df {self.employer_df.shape}\n             Claimant_History_Personal_Details {self.claimant_history_personal_details.shape}\n             Claimant_History_Skill_Details {self.claimant_history_skill_details.shape}")
        self.pi_theft_feature()

        '''
        Frequent Mailing and residential address, bank account and state id change
        '''         
        logger.debug(f"raw_per_df {self.raw_per_df.shape} Skill df {self.skill_df.shape}\n             employment_df {self.employer_df.shape}  employer_df {self.employer_df.shape}\n             Claimant_History_Personal_Details {self.claimant_history_personal_details.shape}\n             Claimant_History_Skill_Details {self.claimant_history_skill_details.shape}")
        self.freq_addr_change()
       
        '''
        Features for suspicious DOB, skill set and education level from historical activities
        '''        
        logger.debug(f"raw_per_df {self.raw_per_df.shape} Skill df {self.skill_df.shape}\n             employment_df {self.employer_df.shape}  employer_df {self.employer_df.shape}\n             Claimant_History_Personal_Details {self.claimant_history_personal_details.shape}\n             Claimant_History_Skill_Details {self.claimant_history_skill_details.shape}")
        self.pi_suspicious()
        
        '''
        Skill related feature engineering
        '''
        logger.debug(f"raw_per_df {self.raw_per_df.shape} Skill df {self.skill_df.shape}\n             employment_df {self.employer_df.shape}  employer_df {self.employer_df.shape}\n             Claimant_History_Personal_Details {self.claimant_history_personal_details.shape}\n             Claimant_History_Skill_Details {self.claimant_history_skill_details.shape}")
        self.skill_features()     
   
        '''
        Employer data preparation
        '''
        logger.debug(f"raw_per_df {self.raw_per_df.shape} Skill df {self.skill_df.shape}\n             employment_df {self.employer_df.shape}  employer_df {self.employer_df.shape}\n             Claimant_History_Personal_Details {self.claimant_history_personal_details.shape}\n             Claimant_History_Skill_Details {self.claimant_history_skill_details.shape}")
        self.employer_features()
        
        '''
        Employment data preparation
        '''
        logger.debug(f"raw_per_df {self.raw_per_df.shape} Skill df {self.skill_df.shape}\n             employment_df {self.employer_df.shape}  employer_df {self.employer_df.shape}\n             Claimant_History_Personal_Details {self.claimant_history_personal_details.shape}\n             Claimant_History_Skill_Details {self.claimant_history_skill_details.shape}")
        claimant_per_emp_sk_df = self.employment_features()
        logger.debug(f"claimant_per_emp_sk_df {claimant_per_emp_sk_df.shape}")
        logger.info("claimant_per_emp_sk_df before required features")
        # logger.info(claimant_per_emp_sk_df)
        logger.debug(f"raw_per_df {self.raw_per_df.shape} Skill df {self.skill_df.shape}\n             employment_df {self.employer_df.shape}  employer_df {self.employer_df.shape}\n             Claimant_History_Personal_Details {self.claimant_history_personal_details.shape}\n             Claimant_History_Skill_Details {self.claimant_history_skill_details.shape}")
        
        '''
        Required features
        '''
        
        document_direct_feature_list = ['CLAIMANT_ID', 'CLAIM_APPLICATION_ID', 'CLAIMANT_MAILING_ADDRESS_CITY',                                         'CLAIMANT_MAILING_ADDRESS_STATE',  'CLAIMANT_RESIDENTIAL_ADDRESS_CITY',                                         'CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_DESC', 'CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID',                                         'CLAIMANT_RESIDENTIAL_ADDRESS_STATE','CLAIMANT_OPTED_FOR_FEDERAL_TAX', 'CLAIMANT_PAYMENT_MODE',                                        "CLAIM_APPLICATION_FILE_DATE","CLAIMANT_OPTED_FOR_STATE_TAX",
                                        'CLAIMANT_SKILL_CODE', 'CLAIAMANT_SKILL_DESCRIPTION', 'CLAIMANT_EDUCATION_LEVEL_CODE',\
                                        'CLAIMANT_EDUCATION_LEVEL_DESC',\

                                        'EMPLOYER_ID','CLAIM_APPLICATION_EMPLOYER_ADDRESS_CITY','CLAIM_APPLICATION_EMPLOYER_STATE',\
                                        'EMPLOYER_INDUSTRY_CODE','EMPLOYER_INDUSTRY_DESC','EMPLOYER_TYPE','EMPLOYER_NAME',\
                                        'JOBTITLE_MENTIONED_BY_EMPLOYER','SEPARATION_REASON_WITH_EMPLOYER']
        if self.is_train == True:
            document_direct_feature_list.append('FRAUD')
        document_derived_feature_list = ['AGE', 'AGE_BUCKET', 'SENIORCITIZEN',  'INVALID_DOB', 'IS_suspicious_DOB',                                         'IS_ID_NOT_LOCAL',  'STATE_ID_THEFT','IS_ID_MISSING','ID_EXPIRED',  'IS_ID_STATE_NOT_MATCHING_MAIL', 'IS_ID_STATE_NOT_MATCHING_RES',                                         'IS_ACC_MISSING','ACC_NUMBER_THEFT', 'IS_BANKING_FRAUD',                                         'IS_MAIL_STATE_NOT_LOCAL', 'IS_RES_STATE_NOT_LOCAL', 'CORRESPONDING_TOGETHER', 'LIVING_TOGETHER', 'RES_ADDR', 'NOT_SAME_MAIL_RES_ADDR', 'NOT_SAME_WORK_PRE_RES_ZIP',                                         'Is_suspicious_skill_history', 'VERY_HIGH_SKILL_COUNT', 'SKILL_COUNT', 'IS_SKILL_PREFERENCE_NOT_MATCHING', 'Is_suspicious_education',                                         'EMP_AGE_BUCKET', 'IS_EMPLOYER_MISSING' ,'WORK_EXPERIENCE_IN_MONTHS','CLAIMANT_MENTIONED_PAY', 'PREF_WORK_EXP','IS_WORK_EXP_AMBIGIOUS', 'is_suspicious_wage_base_period',                                         'EMAIL_THEFT', 'PHONE_THEFT',                                         'MONTH', 'WEEK', 'YEAR', 'YEAR_MONTH', 'YEAR_WEEK','HAS_FRAUD_HISTORY',                                         'IS_FREQ_RES_ADDR_CHANGE','IS_FREQ_MAIL_ADDR_CHANGE','IS_FREQ_BANK_ACC_CHANGE','IS_FREQ_STATE_ID_CHANGE','IS_SUSPICIOUS_EMPLOYER']
        final_list = document_direct_feature_list + document_derived_feature_list

        claimant_per_emp_sk_df.rename(columns={'IS_suspicious_DOB_x':'IS_suspicious_DOB'}, inplace=True)
        
        '''
        We are going to work on this data set for visualization
        '''
        model_df = claimant_per_emp_sk_df[final_list].copy()
        logger.debug(f"model_df {model_df.shape}")
        # List of columns identified for imputation
        fill_cols_m_one = ['ACC_NUMBER_THEFT','IS_suspicious_DOB','ID_EXPIRED','IS_ID_STATE_NOT_MATCHING_RES','IS_ID_STATE_NOT_MATCHING_MAIL',                   'STATE_ID_THEFT','IS_ID_NOT_LOCAL','VERY_HIGH_SKILL_COUNT','IS_SKILL_PREFERENCE_NOT_MATCHING',                   'EMAIL_THEFT','CLAIMANT_OPTED_FOR_STATE_TAX','CLAIMANT_OPTED_FOR_FEDERAL_TAX']
        fill_cols_nd = ['EMP_AGE_BUCKET','EMPLOYER_INDUSTRY_DESC','EMPLOYER_INDUSTRY_CODE','EMPLOYER_ID','CLAIAMANT_SKILL_DESCRIPTION',                        'CLAIMANT_SKILL_CODE','SEPARATION_REASON_WITH_EMPLOYER','CLAIM_APPLICATION_EMPLOYER_ADDRESS_CITY',                         'CLAIM_APPLICATION_EMPLOYER_STATE','CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_DESC',                        'CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID','CLAIMANT_EDUCATION_LEVEL_DESC','CLAIMANT_EDUCATION_LEVEL_CODE',                        'JOBTITLE_MENTIONED_BY_EMPLOYER','EMPLOYER_TYPE','EMPLOYER_NAME'
                        ]
        fill_col_zero = ['SKILL_COUNT','WORK_EXPERIENCE_IN_MONTHS','PREF_WORK_EXP','CLAIMANT_MENTIONED_PAY']
        fill_col_one = ['IS_EMPLOYER_MISSING']
        '''
        Impute missing values and convert all float columns to int and boolean to integer to save space
        '''
        model_df['EMP_AGE_BUCKET'] = model_df['EMP_AGE_BUCKET'].astype(object)
        model_df.update(model_df[fill_cols_m_one].fillna(-1))
        model_df.update(model_df[fill_col_zero].fillna(0))
        model_df.update(model_df[fill_cols_nd].fillna('N/A'))
        model_df.update(model_df[fill_col_one].fillna(1))

        float_cols = model_df.select_dtypes(include='float64').columns
        model_df[float_cols] = model_df[float_cols].astype(int)
        
        
        theft_map = {False:0, True:1, -1:-1}
        bool_cols = ['STATE_ID_THEFT','LIVING_TOGETHER','CORRESPONDING_TOGETHER','PHONE_THEFT','EMAIL_THEFT','ACC_NUMBER_THEFT', 'IS_SKILL_PREFERENCE_NOT_MATCHING']

        model_df[bool_cols] = model_df[bool_cols].applymap(theft_map.get)
        
        fraud_indicator_df = model_df.copy()
        
        yesno_map = {0:'N', 1:'Y', -1:'N/A'}
        yesno_cols = ['CLAIMANT_OPTED_FOR_FEDERAL_TAX','CLAIMANT_OPTED_FOR_STATE_TAX','SENIORCITIZEN','INVALID_DOB',                     'IS_suspicious_DOB','IS_ID_NOT_LOCAL','STATE_ID_THEFT','IS_ID_MISSING','ID_EXPIRED','IS_ID_STATE_NOT_MATCHING_MAIL',                     'IS_ID_STATE_NOT_MATCHING_RES','IS_ACC_MISSING','ACC_NUMBER_THEFT','IS_BANKING_FRAUD','IS_MAIL_STATE_NOT_LOCAL',                     'IS_RES_STATE_NOT_LOCAL','CORRESPONDING_TOGETHER','LIVING_TOGETHER','NOT_SAME_MAIL_RES_ADDR','NOT_SAME_WORK_PRE_RES_ZIP',                     'Is_suspicious_skill_history','VERY_HIGH_SKILL_COUNT','IS_SKILL_PREFERENCE_NOT_MATCHING','Is_suspicious_education',                     'IS_EMPLOYER_MISSING','IS_WORK_EXP_AMBIGIOUS','is_suspicious_wage_base_period','EMAIL_THEFT','PHONE_THEFT','HAS_FRAUD_HISTORY',                     'IS_FREQ_RES_ADDR_CHANGE','IS_FREQ_MAIL_ADDR_CHANGE','IS_FREQ_BANK_ACC_CHANGE','IS_FREQ_STATE_ID_CHANGE','IS_SUSPICIOUS_EMPLOYER']
        fraud_indicator_df[yesno_cols] = fraud_indicator_df[yesno_cols].applymap(yesno_map.get)
        if not self.is_train and self.fraud_indicator_cols:
            try:         
                indicator_df_cols = ["CLAIMANT_ID","CLAIM_APPLICATION_ID"]
                indicator_df_cols.extend(self.fraud_indicator_cols)
                logger.info(f"Fraud indicator columns {indicator_df_cols}")
                fraud_indicator_df = pd.melt(fraud_indicator_df[indicator_df_cols], id_vars=["CLAIMANT_ID","CLAIM_APPLICATION_ID"])
                fraud_indicator_df.rename(columns={"variable":"INDICATOR_NAME", "value":"INDICATOR_VALUE"}, inplace=True)
                fraud_indicator_df_path = os.path.join(self.input_dict["fraud_outputpath"], self.input_dict["fraud_indicator"], "fraud_indicator.csv")
                csv_buffer = StringIO()
                fraud_indicator_df.to_csv(csv_buffer, line_terminator="\n", index=False)
                with self.fs.open(fraud_indicator_df_path, mode="wt", encoding="utf-8") as f:
                    f.write(csv_buffer.getvalue())
                logger.info(f"Fraud indicators file generated at {fraud_indicator_df_path}.")
            except Exception as e:
                traceback.print_exc(limit=100, file=sys.stdout)
                logger.error("Error while generating fraud indicator file", exc_info=True)
        elif not self.is_train and not self.fraud_indicator_cols:
            logger.warning("Fraud indicator column names are missing. Check initialisation of preprocessing class for details.")
        
        '''
        Start of model training data preparation
        '''
        # Step 1: Replace 0 with np.nan
        model_df['CLAIMANT_MENTIONED_PAY'].replace({0:np.nan}, inplace=True)

        # Step 2: Fill the pay using average of following fields
        model_df['CLAIMANT_MENTIONED_PAY'] = model_df['CLAIMANT_MENTIONED_PAY'].fillna(
            model_df.groupby(['EMPLOYER_INDUSTRY_CODE','CLAIMANT_MAILING_ADDRESS_STATE',
                        'CLAIAMANT_SKILL_DESCRIPTION','CLAIMANT_EDUCATION_LEVEL_DESC','AGE'])['CLAIMANT_MENTIONED_PAY'].transform('mean').iloc[0]
        )
        
        return model_df


<<<<<<< HEAD
# In[9]:
=======
# In[13]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


class UIModel():
    def __init__(self, processed_df, input_dict, file_system, training=True):
        logger.info("UIModel init")
        self.input_dict = input_dict
        self.df = processed_df
        self.selected_features = []
        self.is_train = training
        self.fs = file_system
        self.supported_metrics = ['Sensitivity','Specificity','Precision','F1','ROC-AUC']
        self.PCT_CHNG_ALLOWED = 5.0 # Range of percentage change in model metrics
        self.TOP_FEATURES_N = 30 # Count of top features to be selected
    # apply threshold to positive probabilities to create labels
    def to_labels(self, pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')

    def clean_data(self, df_x, drop_cols_list):
        logger.info("clean_data")
        present_cols = [col for col in drop_cols_list if col in df_x.columns]
        df_x  = df_x.drop(present_cols, axis = 1)
        logger.info("{} columns are present".format(present_cols))
        logger.info(f"{list(set(drop_cols_list).difference(set(present_cols)))} columns are being being dropped.")
        return df_x

    # evaluate a model
    def evaluate_model(self, X, y, model,hyper_params,oversample=False):
        logger.info("evaluate_model")
        rs_model = GridSearchCV(model, hyper_params, cv=2, n_jobs = -1)
        if oversample==True:
            method = SMOTE()
            X_over, y_over = method.fit_sample(X, y)
            rs_model.fit(X_over, y_over)
        else:
            rs_model.fit(X, y)

        return rs_model

    # define models to test
    def get_models(self, model_list):
        logger.info("get_models")
        models, names, hyper_params = list(), list(), list()
        # CART
        if 'DecisionTree' in model_list:
            models.append(DecisionTreeClassifier())
            names.append('DecisionTree')
            hyper_params.append({'class_weight':[{0:0.106667, 1:0.89333},None]})

        #LDA
        if 'LDA' in model_list:
            models.append(LinearDiscriminantAnalysis())
            names.append('LDA')
            hyper_params.append({'store_covariance':[True,False]})

        # Bagging
        if 'BaggingClassifier' in model_list:
            models.append(BaggingClassifier(n_estimators=100))
            names.append('BaggingClassifier')
            hyper_params.append({'n_estimators':[10]})
        # RF
        if 'RandomForest' in model_list:    
            models.append(RandomForestClassifier(n_estimators=100))
            names.append('RandomForest')
            hyper_params.append({'class_weight':[{0:0.106667, 1:0.89333},None]})
        # ET
        if 'ExtraTreeClassifier' in model_list:   
            models.append(ExtraTreesClassifier(n_estimators=100))
            names.append('ExtraTreeClassifier')
            hyper_params.append({'n_estimators':[100]})
        #XG
        #scale_pos_weight = total_negative_examples / total_positive_examples
        #For an imbalanced binary classification dataset, the negative class refers to the majority class (class 0) and the positive class refers to the minority class (class 1).
        #359958/42146
        if 'XGBoostClassifier' in model_list:   
            models.append(XGBClassifier(booster='gbtree', n_jobs=-1))
            names.append('XGBoostClassifier')
            hyper_params.append({'scale_pos_weight':[1, 5.08]})

        return models, names, hyper_params

    #function to use for scoring 
    def get_classification_metrics(self, y_test,preds):
        logger.info("get_classification_metrics")
        TN, FP, FN, TP = confusion_matrix(y_test, preds).ravel()

        sensi = TP/(TP + FN)
        speci= TN/(TN + FP)
        preci = TP/(TP + FP)
        f1= 2*(preci*sensi)/(preci+sensi)
        return round(sensi,3),round(speci,3),round(preci,3),round(f1,3)

    def scores(self, t, name, x_train, x_test, y_train, y_test):
        logger.info("scores")

        to_labels = self.to_labels
        
        default_metric = []
        optimized_metric = []
        logger.info(f"{name} classification metric")
        train_score = round(t.score(x_train, y_train),3)
        test_score =  round(t.score(x_test, y_test),3)

        default_metric.append(train_score)
        default_metric.append(test_score)
        optimized_metric.append(train_score)
        optimized_metric.append(test_score)    
        #Evaluation metrics
        predictions = t.predict(x_test)
        sensi, speci,preci,f1 = self.get_classification_metrics(y_test, predictions)

        default_metric.append(round(sensi,3))
        default_metric.append(round(speci,3))
        default_metric.append(round(preci,3))
        default_metric.append(round(f1,3))

        pred_proba = [i[1] for i in t.predict_proba(x_test)]
        auc_score = roc_auc_score(y_test, pred_proba)
        default_metric.append(round(auc_score,3))


        logger.info("***********Result after threshold tuning***********")
        #Section below is for threshold tuning
        # predict probabilities
        yhat = t.predict_proba(x_test)
        # keep probabilities for the positive outcome only
        probs = yhat[:, 1]
        # define thresholds
        thresholds = np.arange(0, 1, 0.01)
        # evaluate each threshold
        pred_scores = [f1_score(y_test, to_labels(probs, t)) for t in thresholds]
        # get best threshold
        ix = np.argmax(pred_scores)
        logger.info('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], pred_scores[ix]))

        threshold = thresholds[ix]

        predicted = t.predict_proba(x_test)
        predicted[:,0] = (predicted[:,0] < threshold).astype('int')
        predicted[:,1] = (predicted[:,1] >= threshold).astype('int')


        #Evaluation metrics
        #predictions = t.predict(x_test)
        sensi, speci,preci,f1 = self.get_classification_metrics(y_test, predicted[:,1])

        optimized_metric.append(round(sensi,3))
        optimized_metric.append(round(speci,3))
        optimized_metric.append(round(preci,3))
        optimized_metric.append(round(f1,3))

        auc_score = roc_auc_score(y_test, predicted[:,1])
        optimized_metric.append(round(auc_score,3))
        return default_metric, optimized_metric, threshold
    
    def limit_records_for_training(self, _date):
        logger.info("limit_records_for_training")
        '''
        ENABLE THE CELL BELOW TO LIMIT THE DATA FROM MAY-2020 ONWARDS (2020-05-01), i.e. last 6 months 
        '''
        
            
        self.df["CLAIM_APPLICATION_FILE_DATE"] = pd.to_datetime(self.df["CLAIM_APPLICATION_FILE_DATE"])

        
        #the records only need to be limited for training and not prediction
        if self.is_train == True:
            self.df = self.df[self.df['CLAIM_APPLICATION_FILE_DATE']>=_date]            
        cols_to_be_dropped = ['NOT_SAME_WORK_PRE_RES_ZIP','RES_ADDR','MONTH', 'WEEK', 'YEAR', 'YEAR_MONTH', 'YEAR_WEEK',                          'CLAIM_APPLICATION_FILE_DATE', 'AGE','CLAIAMANT_SKILL_DESCRIPTION','CLAIMANT_EDUCATION_LEVEL_DESC',                          'CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_DESC','EMPLOYER_INDUSTRY_DESC','JOBTITLE_MENTIONED_BY_EMPLOYER',                              'EMPLOYER_NAME', 'CLAIMANT_EDUCATION_LEVEL_CODE']
        self.df = self.clean_data(self.df,cols_to_be_dropped)
            
    def data_transformation(self):
        logger.info("data_transformation")
        cols_to_removed = ['IS_BANKING_FRAUD','IS_FREQ_RES_ADDR_CHANGE','IS_FREQ_MAIL_ADDR_CHANGE','IS_FREQ_BANK_ACC_CHANGE',                   'IS_FREQ_STATE_ID_CHANGE']
        self.df = self.clean_data(self.df,cols_to_removed)
        
        '''
        Categorical encoding
        '''
        #, 'WORK_EXPERIENCE_IN_MONTHS','AGE', 
        cont_var = ['PREF_WORK_EXP', 'SKILL_COUNT','CLAIMANT_MENTIONED_PAY','WORK_EXPERIENCE_IN_MONTHS']
        nom_var = ['AGE_BUCKET', 'CLAIMANT_MAILING_ADDRESS_CITY', 'CLAIMANT_MAILING_ADDRESS_STATE', 'CLAIMANT_PAYMENT_MODE', 'CLAIMANT_RESIDENTIAL_ADDRESS_CITY',
        'CLAIMANT_RESIDENTIAL_ADDRESS_COUNTY_ID', 'CLAIMANT_RESIDENTIAL_ADDRESS_STATE', 'CLAIMANT_SKILL_CODE', 'CLAIM_APPLICATION_EMPLOYER_ADDRESS_CITY', 'CLAIM_APPLICATION_EMPLOYER_STATE',
        'EMPLOYER_INDUSTRY_CODE', 'EMP_AGE_BUCKET', 'SEPARATION_REASON_WITH_EMPLOYER']
        
        nom_var_object_type = self.df[self.df.columns.difference(["CLAIMANT_ID","CLAIM_APPLICATION_ID"])].select_dtypes(include='object').columns.tolist()
        nom_var = nom_var + nom_var_object_type
        nom_var = list(set(nom_var)) 

        if (self.is_train == True):
            X = self.df.drop(["FRAUD", "CLAIMANT_ID","CLAIM_APPLICATION_ID"], axis=1)
            logger.debug(X.columns)
            y = self.df["FRAUD"].values
            cat_enc = MEstimateEncoder(cols=nom_var)
            cat_enc.fit(X, y)
            save_model(cat_enc, "cat_transformation_ins.pkl")
            '''
            Scaling
            '''
            scaler = MinMaxScaler()
            scaler.fit(X[cont_var])
            save_model(scaler, "min_max_transformation.pkl")
            df_encoded = cat_enc.transform(X)
            df_encoded["FRAUD"] = y
        else:
            #TODO: comment on why the encoding was not done from outside
            X = self.df.drop(["CLAIMANT_ID","CLAIM_APPLICATION_ID"], axis=1)
            logger.debug(X.columns)
            cat_enc = load_model("cat_transformation_ins.pkl")
            scaler = load_model("min_max_transformation.pkl")
            df_encoded = cat_enc.transform(X)
        df_encoded.index = list(self.df["CLAIM_APPLICATION_ID"])
        df_encoded[cont_var] = scaler.transform(df_encoded[cont_var])
        #TODO: Uncomment the below line after the NaN error got fixed.
        df_encoded.fillna(0, inplace=True)
        self.df = df_encoded
        
    
    def select_model_features(self):
        logger.info("select_model_features")
        self.check_for_retraining = True if (self.input_dict.get("check_for_retraining", "true") == "true") else False
        # Check if the model folder already has saved features from earlier training.
        saved_features = None
        try:
            saved_features = load_model("selected_features.pkl")
        except:
            logger.warning("No saved features file found from previous training.")
            logger.warning("Hence check_for_retraining flag set to FALSE.")
            self.check_for_retraining = False
        # First time training and not retraining.
        if self.is_train and not self.check_for_retraining:
            logger.info("First time training and not checking for retraining")
            transformed_df = self.df
            # Added forward fill for missing values
            transformed_df = transformed_df.fillna(method='ffill')
            x = transformed_df.drop(['FRAUD'], axis=1)
            y = transformed_df['FRAUD']
            features = x
            labels = y
           # 3. Variable Selection

            # Feature selection methods

            ## 3.1 WOE and IV
            '''
            #to be used later, for now throwing error
            final_iv, IV = data_vars(df_final[df_final.columns.difference(['FRAUD'])],df_final.FRAUD)

            IV = IV.rename(columns={'VAR_NAME':'index'})

            '''
    #         IV.sort_values(['IV'],ascending=0)

            ## 3.2 Variable Importance using Random Forest 
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(features,labels)
            VI = pd.DataFrame(clf.feature_importances_, columns = ["RF"], index=features.columns)

            ## 3.3 Recursive Feature Elimination
            model = LogisticRegression(solver="lbfgs", max_iter=999)
            rfe = RFE(model, n_features_to_select=20)
            fit = rfe.fit(features, labels)
            selected = pd.DataFrame(rfe.support_, columns = ["RFE"], index=features.columns)
            ## 3.4 Variable Importance using Extratrees Classifier 
            model = ExtraTreesClassifier(n_estimators=100)
            model.fit(features, labels)
            FI = pd.DataFrame(model.feature_importances_, columns = ["Extratrees"], index=features.columns)
            ## 3.5 Chi Square
            model = SelectKBest(score_func=chi2, k=5)
            fit = model.fit(features.abs(), labels)
            pd.options.display.float_format = '{:.2f}'.format
            chi_sq = pd.DataFrame(fit.scores_, columns = ["Chi_Square"], index=features.columns)
            ## 3.6 L1 feature selection
            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels)
            model = SelectFromModel(lsvc,prefit=True)
            l1 = pd.DataFrame(model.get_support(), columns = ["L1"], index=features.columns)
            ## 3.7 Combine all methods together 
            #Selected,
            dfs = [VI,  selected, FI, chi_sq, l1]
            final_results = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), dfs)
            logger.debug(f"final results \n{final_results}")
            ## 3.8 Vote each variable
            columns = ['RF', 'RFE','Extratrees', 'Chi_Square','L1']
            score_table = pd.DataFrame(index=final_results.index)
            for i in columns:
                score_table[i] = final_results.index.isin(final_results.nlargest(5,i).index.tolist()).astype(int)
            logger.debug(f"Score_table {score_table}")
            score_table['RFE'] = final_results['RFE'].astype(int)
            score_table['L1'] = final_results['L1'].astype(int)
            score_table['final_score'] = score_table.sum(axis=1)
            ## 3.8 Multicollinearity check using VIF - To reduce dimension
            selected_feature_cols = score_table.nlargest(self.TOP_FEATURES_N,'final_score').index.tolist()
            save_model(selected_feature_cols, "selected_features.pkl")
            final_vars = selected_feature_cols + ['FRAUD']
        elif self.is_train and self.check_for_retraining:
            logger.info("Checking for model retraining")
            final_vars = saved_features + ['FRAUD']
        else:
            logger.info("Loading selected features from previous training for model scoring(prediction).")
            final_vars = load_model("selected_features.pkl")
            
        logger.info(f"Selected features are {final_vars}")
        self.selected_features = final_vars
        
    def split_data_after_feature_selection(self, selected_features, dataframe):
        dataframe = dataframe[selected_features].fillna(0)
        dataframe.dropna(inplace=True)
        X = dataframe.drop(['FRAUD'], axis=1)
        label = dataframe['FRAUD']
        x_train, x_test, label_train, label_test = train_test_split(X, label, stratify=label, test_size=0.25)
        return x_train, x_test, label_train, label_test

    def modelling(self, x_train, x_test, y_train, y_test, metric:str="Sensitivity"):
        """
        Finds the best model based on the given metric.

        Parameters:
        ---
        metric: Name of the metric based on which models will be compared. Default value is 'Sensitivity'.
                Supported metric names are Sensitivity, Specificity, Precision, F1, ROC-AUC
        """
        logger.info("modelling")
        '''
        Prepare the data to summarize results from all classifiers
        '''
        model_names = ['DecisionTree','LDA','BaggingClassifier','RandomForest','ExtraTreeClassifier','XGBoostClassifier']
        model_names = model_names[1:]
        # define models
        models, names,hyper_params = self.get_models(model_names)

        level0_idx = ['Default', 'Optimized']
        level1_idx = ['Model','Threshold']
        cols = ['Train_score', 'Test_Score','Sensitivity','Specificity','Precision','F1','ROC-AUC']
        multi_index = pd.MultiIndex.from_product([model_names,level0_idx], names=level1_idx)
        df_score = pd.DataFrame(data='',index=multi_index,columns=cols)
        
        # evaluate each model
        for i in range(len(models)):
            # evaluate the model and store results
            oversampling = False
            if names[i] == "RandomForest":
                oversampling = True
            clf = self.evaluate_model(x_train, y_train, models[i], hyper_params[i], oversampling)
            d_metric, o_metric, threshold = self.scores(clf, names[i], x_train, x_test, y_train, y_test)

            df_score.loc[(names[i],'Default'),:] = d_metric
            df_score.loc[(names[i],'Optimized'),:] = o_metric
        # Get the best model name based on the metric. Combined model rows are ignored from the df_score.
        if metric not in self.supported_metrics:
            raise ValueError(f"Currently {metric} metric is not supported for model comparison.")
        metric_series = df_score[metric].astype("float64")
        best_metric_index = metric_series.idxmax()
        model_name = best_metric_index[0]
        logger.info(f"Scores for all the models evaluated are\n{df_score}")
        return model_name
        
    def xgboost_model(self, x_train, x_test, y_train, y_test, metric):
        logger.info("xgboost_model")
        '''
        10 fold cross validation and Hyper parameter Tuning for Weighted XGBoost
        '''
        xg = XGBClassifier(booster='gbtree', n_jobs=-1, use_label_encoder=False)
        xg_values = {'max_depth': [3, 4, 5, 6],
                     'eta': [0.05, 0.1, 0.15, 0.3],
                     'reg_lambda': [0.01, 0.05, 0.1, 0.5, 1],
                     'reg_alpha': [0.01, 0.05, 0.1, 0.5, 1],
                     'gamma': [0, 1, 2, 3],
                     'n_estimators': [150, 250, 350, 450, 500, 550, 600, 650],
                     'scale_pos_weight':[1, 5.08],
                      }
        rs_xg = RandomizedSearchCV(xg, xg_values, cv=2, n_jobs = -1, random_state=42)
        rs_xg.fit(x_train, y_train)
        logger.info(f"Best parameters for xg boost are {rs_xg.best_params_}")
        self.best_model = rs_xg.best_estimator_
        save_model(rs_xg.best_estimator_, "model_best.pkl")
        cols = ['Train_score', 'Test_Score','Sensitivity','Specificity','Precision','F1','ROC-AUC']
        df_score = pd.DataFrame(index=["Default","Optimized"], columns=cols)
        d_metric, o_metric, threshold = self.scores(rs_xg.best_estimator_, "XGBClassifier", x_train, x_test, y_train, y_test)
        df_score.loc["Default",:] = d_metric
        df_score.loc["Optimized",:] = o_metric
        logger.info(f"Scores for XGBClassifier are\n{df_score}")
        metric_series = df_score[metric].astype("float64")
        best_metric_index = metric_series.idxmax()
        metric_value = metric_series.loc[best_metric_index]
        # Save best models metric value which can be used for comparison during retraining.
        save_model({"metric":metric, "value":metric_value, "opt_threshold": threshold}, "best_model_metrics.pkl")
        '''
        Final Model: Weighted XGBoost
        
        The final fitted model is the weighted XGBoost on the last 6 months dataset with no oversampling. The best estimators of the model are as follows:

            Scale_pos_weight: 1,
            Reg_lambda (L2 regularization weight): 0.01,
            Reg_alpha (L1 regularization weight): 0.01,
            N_estimators: 250,
            Max_depth: 5,
            Gamma: 2,
            Eta: 0.15
        '''
        logger.info("Final Model: Weighted XGBoost")
        logger.info(f"train score: {round(rs_xg.best_estimator_.score(x_train, y_train),3)}")
        logger.info(f"test score: {round(rs_xg.best_estimator_.score(x_test, y_test),3)}")
        
        
        '''
        Final fitted model's performance
        The model had a training accuracy score of 0.967 and a test accuracy of 0.844. The high accuracy score hint of a low bias (it is only a hint as accuracy is not a good measure of bias in imbalance class problems). An accuracy score difference of 0.123 between train and test is relatively small. Thus, this model can be said to have low variance and is generalizable on unseen data.

            The number of cases for each class of the test set is shown in the confusion matrix below. The y-axis shows the actual classes while the x-axis shows the predicted classes.

            True negative refers to non-fraud cases that are classified as non-fraud cases (161 cases, which makes up 64.40% of the test set's size).
            True positive refers to fraud cases that are correctly classified as fraud cases (50 cases, which makes up 20.00% of the test set's size).
            False negative are fraud cases that are classified as non-fraud cases (12 cases, which makes up 4.80% of the test set's size).
            False positive are non-fraud cases that are classified as fraud cases (27 cases, which makes up 10.80% of the test set's size).
            Percentage out of total sample size of the test set is printed on each quadrant.
        '''
        #confusion matrix
        predictions = rs_xg.best_estimator_.predict(x_test)
        try:
            # P.S. get_ipython will be set in global context, when running from ipython shell or jupyter notebook.
            # Hence it will not raise exception when run from jupyter, otherwise exception will be raised.
            # This is done to disable plotting code when model is running from normal python shell. 
            get_ipython
            cf_matrix = confusion_matrix(y_test, predictions)

            #labels for the inside of heatmap
            group_names = ['True Neg','False Pos','False Neg','True Pos']
            group_counts = ['n={0:0.0f}'.format(value) for value in cf_matrix.flatten()]
            group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

            #put them next line
            labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

            #in array 2,2 cos the heatmap will be 2,2
            labels = np.asarray(labels).reshape(2,2)

            #class labeling
            yticklabels=['Not Fraud','Fraud']
            xticklabels=['Predicted as\nNot Fraud','Predicted as\nFraud']


            # Set the default matplotlib figure size to 7x7:
            fix, ax = plt.subplots(figsize=(6,5))

            # Plot the heatmap with seaborn.
            # Assign the matplotlib axis the function returns. This will let us resize the labels.
            sns.set()
            ax = sns.heatmap(cf_matrix, annot=labels, 
                        xticklabels = xticklabels, yticklabels = yticklabels, 
                        fmt='', cmap='Blues')

            # Resize the labels.
            ax.set_title('Confusion matrix', fontsize=15,  fontweight='bold')
            ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=11, ha= 'center', rotation=0 )
            ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=11, va="center", rotation=0)
            
            logger.info("Result after threshold tuning")
            #Section below is for threshold tuning
            # predict probabilities
            yhat = rs_xg.best_estimator_.predict_proba(x_test)
            # keep probabilities for the positive outcome only
            probs = yhat[:, 1]
            # define thresholds
            thresholds = np.arange(0, 1, 0.01)
            # evaluate each threshold
            pred_scores = [f1_score(y_test, self.to_labels(probs, t)) for t in thresholds]
            # get best threshold
            ix = np.argmax(pred_scores)

            logger.info('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], pred_scores[ix]))
            
            threshold = thresholds[ix]

            predicted = rs_xg.best_estimator_.predict_proba(x_test)
            predicted[:,0] = (predicted[:,0] < threshold).astype('int')
            predicted[:,1] = (predicted[:,1] >= threshold).astype('int')

            #confusion matrix
            #predictions = rs_xg.best_estimator_.predict(x_test)
            cf_matrix = confusion_matrix(y_test, predicted[:,1])

            #labels for the inside of heatmap
            group_names = ['True Neg','False Pos','False Neg','True Pos']
            group_counts = ['n={0:0.0f}'.format(value) for value in cf_matrix.flatten()]
            group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

            #put them next line
            labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

            #in array 2,2 cos the heatmap will be 2,2
            labels = np.asarray(labels).reshape(2,2)

            #class labeling
            yticklabels=['Not Fraud','Fraud']
            xticklabels=['Predicted as\nNot Fraud','Predicted as\nFraud']

            # Set the default matplotlib figure size to 7x7:
            fix, ax = plt.subplots(figsize=(6,5))

            # Plot the heatmap with seaborn.
            # Assign the matplotlib axis the function returns. This will let us resize the labels.
            sns.set()
            ax = sns.heatmap(cf_matrix, annot=labels, 
                        xticklabels = xticklabels, yticklabels = yticklabels, 
                        fmt='', cmap='Blues')

            # Resize the labels.
            ax.set_title('Confusion matrix', fontsize=15,  fontweight='bold')
            ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=11, ha= 'center', rotation=0 )
            ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=11, va="center", rotation=0)
        except NameError as ne:
            # Catch the NameError generated from get_ipython
            logger.info("Notebook dependant code disabled as it is not running from notebook.")
        logger.info(classification_report(y_test, predictions, target_names=['Not Fraud',"Fraud"]))
        
    def randomforest_model(self, x_train, x_test, y_train, y_test, metric):
        logger.info("randomforest_model")
        rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
        method = SMOTE()
        x_over_sample, y_over_sample = method.fit_resample(x_train, y_train)
        #rf.fit(x_train, y_train)
        rf.fit(x_over_sample, y_over_sample)
        self.best_model = rf
        save_model(rf, "model_best.pkl")
        cols = ['Train_score', 'Test_Score','Sensitivity','Specificity','Precision','F1','ROC-AUC']
        df_score = pd.DataFrame(index=["Default","Optimized"], columns=cols)
        d_metric, o_metric, threshold = self.scores(rf, "RandomForest", x_train, x_test, y_train, y_test)
        df_score.loc["Default",:] = d_metric
        df_score.loc["Optimized",:] = o_metric
        logger.info(f"Scores for RandomForestClassifier are\n{df_score}")
        metric_series = df_score[metric].astype("float64")
        best_metric_index = metric_series.idxmax()
        metric_value = metric_series.loc[best_metric_index]
        # Save best models metric value which can be used for comparison during retraining.
        save_model({"metric":metric, "value":metric_value, "opt_threshold": threshold}, "best_model_metrics.pkl")
        
    def predict_output(self, score_df, threshold):
        ''' Load the model from pickle file and use it for prediction.
            Also to save time, the loaded model is saved in best_model property for future use.
        '''
        logger.info("predict_output")
        model_best = load_model("model_best.pkl")
        self.best_model = model_best
        rf_probas = model_best.predict_proba(score_df)
        logger.info(f"Probabilities ndarray shape:{rf_probas.shape}")
        rf_probas_fraud = np.round(rf_probas[:,1]*100, decimals=2)
        rf_predictions = (rf_probas_fraud > threshold).astype('int')
        return rf_predictions, rf_probas_fraud
    
    def model_explainability(self, model_feature_names=None):
        rs_xg = load_model("model_best.pkl")
        logger.info(f"Best model {type(rs_xg)}")
        feature_important= None
        if isinstance(rs_xg,XGBClassifier):
            feature_important= rs_xg.get_booster().get_score(importance_type='weight')
        elif isinstance(rs_xg,RandomForestClassifier):
            feature_important= dict(zip(model_feature_names, rs_xg.feature_importances_))
        glb_ftr_imp_df = pd.DataFrame([feature_important]).T
        glb_ftr_imp_df.reset_index(inplace=True)
        glb_ftr_imp_df.columns = ["feature", "value"]
        path_glb_ftr_imp = os.path.join(self.input_dict["fraud_outputpath"], self.input_dict["feature_importance"], "global_feature_importance.csv")
        csv_buffer = StringIO()
        glb_ftr_imp_df.to_csv(csv_buffer, line_terminator="\n", index=False)
        with self.fs.open(path_glb_ftr_imp, mode="wt", encoding="utf-8") as f:
            f.write(csv_buffer.getvalue())
        logger.info(f"Global feature importance file generated at {path_glb_ftr_imp}.")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        if isinstance(rs_xg,XGBClassifier):
            plot_importance(rs_xg, importance_type='weight', 
                            max_num_features=20, height=0.4, ax=ax, xlabel='Num of times feature used to split data across all trees',
                        color='lightblue')
        
    def compare_model_by_metric(self, metric_name:str, metric_value, val_x, val_label, desired_direction:str="high") -> bool:
        """
        Comapre metrics given by model on new data set with metrics stored for the previous data set.
        If metrics value deteroits w.r.t desired_direction on new data, then return True else return False.

        :param metric_name
            Name of the metric based on which comparison will be done.
        :param metric_value
            Metric value stored previously for the old data.        
        :param val_x
            New dataset on which training will be done
        :param val_label
            The target label of the new dataset.
        :param desired_direction default="high"
            If set to "high" then higher value of metric is better, if set to "low" then lower value of metric
            is better for model.
        """
        pretrained_model = load_model("model_best.pkl")
        val_probas = pretrained_model.predict_proba(val_x)
        # 1. Calculate ROC-AUC score        
        probs = val_probas[:, 1] # keep probabilities for the positive outcome only
        thresholds = np.arange(0, 1, 0.01) # define thresholds
        # evaluate each threshold
        pred_scores = [f1_score(val_label, self.to_labels(probs, t)) for t in thresholds]
        ix = np.argmax(pred_scores)
        logger.info('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], pred_scores[ix]))
        threshold = thresholds[ix]
        val_probas[:,0] = (val_probas[:,0] < threshold).astype('int')
        val_probas[:,1] = (val_probas[:,1] >= threshold).astype('int')
        roc_auc_score_ = roc_auc_score(val_label, val_probas[:,1])
        # 2. Calculate sensitivity, specificity, precision and f1 score.
        sensi, speci, preci, f1 = self.get_classification_metrics(val_label, val_probas[:,1])
        metrics_dict = {'Sensitivity':sensi, 'Specificity':speci, 'Precision':preci, 'F1':f1, 'ROC-AUC':roc_auc_score_}
        new_metric_value = metrics_dict.get(metric_name)
        logger.info(f"New metric {metric_name} value is {new_metric_value}")
        abs_diff = abs(metric_value - new_metric_value)
        pct_chng = 100*abs_diff/metric_value
        logger.info(f"Percentage change is {pct_chng}")
        if metric_name not in self.supported_metrics:
            raise ValueError(f"Currently {metric_name} metric is not supported for model comparison.")
        if desired_direction == "high":
            if metric_value > new_metric_value and (pct_chng > self.PCT_CHNG_ALLOWED):
                return True
            else:
                return False
        elif desired_direction == "low":
            if metric_value < new_metric_value and (pct_chng > self.PCT_CHNG_ALLOWED):
                return True
            else:
                return False

    def is_retraining_required(self, metric_name, desired_direction, val_x, val_label):
        training_required = True
        try:
            old_metrics = load_model("best_model_metrics.pkl")
            old_metric_name = old_metrics.get("metric")
            metric_value = old_metrics.get("value")
            logger.info(f"Old metric {old_metric_name} value is {metric_value}")
            if old_metric_name != metric_name:
                raise ValueError(f"Old metric name {old_metric_name} doesnt match with {metric_name}.")
            training_required = self.compare_model_by_metric(metric_name, metric_value, val_x, val_label, desired_direction=desired_direction)
        except FileNotFoundError as fntfe:
            logger.error("Best Model metrics file not found.", exc_info=1)
        except Exception as e:
            logger.error("Exception during check for model retraining.", exc_info=1)
        finally:
            return training_required


# ### Build Model 
# * 			The methods `train_model(inputdata:dict)` and `get_prediction(inputdata:dict)` are mandatory and needs to be present in the `custom.py` file. 
# * 			The parameter `inputdata` in both `train_model()` and `get_prediction()` needs to be a dictionary. 
# 
#  			For Example: 
#  `inputdata= { "outputpath" : "**/output/",  "inputfilepath" : "filepath" }`

<<<<<<< HEAD
# In[10]:
=======
# In[14]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


input_dict_train = {
        "file_path" : "/u01/HomeDir/mdesuser/POC_Data",
        "raw_per_df" : "claimant_personal_details_actual",
        "interim_df" : "claimant_personal_details_delta", # This is optional,may or may not be present in location
        "skill_df" : "claimant_skill_details",
        "employment_df": "claimant_employment",
        "employer_df": "employers_master_details",
        "Claimant_History_Personal_Details" : "claimant_history_personal_details",
        "Claimant_History_Skill_Details" : "claimant_history_skill_details",
        "start_date" : "2020-05-01",
        "check_for_retraining" : True,
        "comparison_metric" : "F1", # Check for metric is missing
        "fraud_outputpath" : "/u01/HomeDir/mdesuser/POC_Data/MODEL_OUTPUT",
        "feature_importance": "feature_importance",
        "fraud_indicator": "Fraud_Indicator",
        "model_feature_importance": "Model_Feature_Importance",
        "model_output_load": "Model_Output_Load",
        "outputpath" : "/u01/HomeDir/mdesuser/POC_Data"
    }


<<<<<<< HEAD
# In[11]:
=======
# In[15]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


def train_model(inputdata: Dict):
    '''This method is used for building and training the Model.'''
    logger.info("Unemployment insurance fraud model training")
    logger.info(f"The input parameters for training are:\n{inputdata}")
    preprocessed_data = None
    file_system = fsspec.filesystem("file") # Currently using local file system protocol
    try:        
        preprocess_train = Preprocessing(file_system, inputdata)
        preprocessed_data = preprocess_train.data_preparation()
    except Exception as e:
        traceback.print_exc(limit=100, file=sys.stdout)
        logger.error("Exception during preprocessing of data!", exc_info=True)
        sys.exit("Exception during data preprocessing during training!")
    # Call to python garbage collector. This call doesnt gaurenttee garbage memory will be collected.
    gc.collect()
    logger.info("Preprocessing done. Model training will start now.")
    model_train = None
    try:
        model_train = UIModel(preprocessed_data, inputdata, file_system)
        # Limit data for last 6 months. Also drop some unnecessary columns
        model_train.limit_records_for_training(pd.to_datetime(inputdata["start_date"], format="%Y-%m-%d"))
        # transformation for ML
        model_train.data_transformation()
        model_train.select_model_features()
        best_model = None
        x_train, x_test, label_train, label_test = model_train.split_data_after_feature_selection(model_train.selected_features, model_train.df)
        # make the check for retraining optional parameter given by user.
        check_for_retraining_from_input = True if (inputdata.get("check_for_retraining", "true") == "true") else False
        check_for_retraining = check_for_retraining_from_input and model_train.check_for_retraining
        # make the comparison metric an optional parameter given by user.
        comparison_metric = inputdata.get("comparison_metric","Sensitivity")
        if check_for_retraining:
            logger.info("Check for model retraining on new data.")      
            # default desired direction is high. Meaning higher the metric value, the better is the model.
            desired_direction = inputdata.get("desired_direction", "high")
            is_training_required = model_train.is_retraining_required(comparison_metric, desired_direction, x_test, label_test)
            if is_training_required:
                # If no exception occurred while checking for retraining and retraining is indeed needed !
                logger.warning(f"RETRAINING OF MODEL IS NEEDED AS METRIC ``{comparison_metric}`` IS DRIFTING.")
            else:
                # If no exception occured while checking for retraining and retraining is not needed.                    
                    # If no exception occured while checking for retraining and retraining is not needed.                    
                # If no exception occured while checking for retraining and retraining is not needed.                    
                logger.info(f"PRETRAINED MODEL IS FIT FOR CURRENT DATA.")
                # In this case training function will stop, as retraining is not needed.
                return
            # elif err:
            #     # Some error occurred while checking for retraining.
            #     # Error may occur:
            #     #  1) if no pretrained model is present, meaning this is the first time training is happening.
            #     #  2) Due to some unforeseen condition not handled in `is_retraining_required` method.
            #     logger.warning(f"Evaluation of previous model based on metrics failed. Normal model training will start.")
            
        best_model = model_train.modelling(x_train, x_test, label_train, label_test, metric=comparison_metric)
        logger.info(f"Best model selected after CV based on {comparison_metric} metric is {best_model}")
        if best_model =="XGBoostClassifier":            
            model_train.xgboost_model(x_train, x_test, label_train, label_test, comparison_metric)
            model_train.model_explainability()
        elif best_model=="RandomForest":
            model_train.randomforest_model(x_train, x_test, label_train, label_test, comparison_metric)
            model_train.model_explainability(model_feature_names=x_train.columns)
    except Exception as e:
        traceback.print_exc(limit=100, file=sys.stdout)
        logger.error("Exception during training of model!", exc_info=True)
        sys.exit("Exception during model training!")


# In[ ]:


train_model(input_dict_train)


<<<<<<< HEAD
# In[12]:
=======
# In[19]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


def get_prediction(inputdata: Dict):
    '''This method is used for predicting the output when appropriate data is fed to the trained model.'''
    logger.info("Unemployment insurance fraud prediction")
    logger.info(f"The input parameters for prediction are:\n{inputdata}")
    inputdata["history_data_start_date"] = pd.to_datetime(inputdata["history_data_start_date"], format="%Y-%m-%d")
    preprocessed_data = None
    file_system = fsspec.filesystem("file") # Currently using local file system protocol
    try:
        preprocess_scoring = Preprocessing(file_system, inputdata, training=False)
        preprocessed_data = preprocess_scoring.data_preparation()
    except Exception as e:
        traceback.print_exc(limit=100, file=sys.stdout)
        logger.error("Exception during preprocessing of data!", exc_info=True)
        sys.exit("Exception during data preprocessing during prediction!")
    logger.info("Preprocessing done. Model scoring will start now.")
    outputpath = inputdata["fraud_outputpath"]
    model_scoring = None
    try:
        model_scoring = UIModel(preprocessed_data, inputdata, file_system, training=False)
        model_scoring.limit_records_for_training(pd.to_datetime(inputdata["start_date"], format="%Y-%m-%d"))
        model_scoring.data_transformation()
        model_scoring.select_model_features()

        score_df = model_scoring.df[model_scoring.selected_features].fillna(0)
        score_df_with_ids = score_df.merge(preprocessed_data[["CLAIMANT_ID", "CLAIM_APPLICATION_ID"]], left_index=True, right_on=["CLAIM_APPLICATION_ID"])        
        best_model_metrics = load_model("best_model_metrics.pkl")
        threshold = best_model_metrics.get("opt_threshold")
        if inputdata.get("user_defined_threshold", 0):
            threshold = float(inputdata.get("user_defined_threshold"))
            threshold = round(threshold*100, 2)
        logger.info(f"Threshold used {threshold}")
        predictions, fraud_probas = model_scoring.predict_output(score_df, threshold)

        score_df_with_ids["FRAUD"] = predictions.tolist()
        score_df_with_ids["PROBABILITY_SCORE"] = fraud_probas.tolist()
        fraud_map = {0:'N', 1:'Y'}
        score_df_with_ids['FRAUD'] = score_df_with_ids['FRAUD'].apply(fraud_map.get)
        fraud_probability_df = score_df_with_ids[["CLAIMANT_ID", "CLAIM_APPLICATION_ID", "FRAUD", "PROBABILITY_SCORE"]]
        fraud_probability_df = fraud_probability_df.rename(columns={"FRAUD":"IS_FRAUD_REPORTED"})
        fraud_probability_df_path = os.path.join(outputpath, inputdata["model_output_load"], "fraud_probability.csv")
        csv_buffer = StringIO()
        fraud_probability_df.to_csv(csv_buffer, line_terminator="\n", index=False)
        with file_system.open(fraud_probability_df_path, mode="wt", encoding="utf-8") as f:
            f.write(csv_buffer.getvalue())
        logger.info(f"Fraud probability file saved at {fraud_probability_df_path}.")
        # SHAP value calculation file preparation
        temp_df = fraud_probability_df.merge(score_df_with_ids, on=['CLAIMANT_ID', 'CLAIM_APPLICATION_ID'], how='inner')
        logger.info(f"temp_df shape {temp_df.shape}")
        features_selected = model_scoring.selected_features
        logger.info(f"Length of features selected {len(features_selected)}\n{features_selected}")
        
        clf = model_scoring.best_model
        # Create object that can calculate shap values
        explainer = shap.TreeExplainer(clf)
        # Model input file
        shap_values = [explainer.shap_values(row[2].reshape(1,-1))[0].reshape(1,-1).flatten().tolist()+[row[0], row[1]] for row in zip(temp_df['CLAIMANT_ID'],temp_df['CLAIM_APPLICATION_ID'],temp_df[features_selected].values)]
        logger.info(f"Length of shap_values is {len(shap_values)}")
        # Create a dataframe to store shap value
        _columns = features_selected + ['CLAIMANT_ID', 'CLAIM_APPLICATION_ID']
        shap_df = pd.DataFrame(shap_values, columns=_columns)
        logger.info(f"Shap_df shape {shap_df.shape}")
        shap_df['CLAIMANT_ID'] = shap_df['CLAIMANT_ID'].astype(int)
        shap_df['CLAIM_APPLICATION_ID'] = shap_df['CLAIM_APPLICATION_ID'].astype(int)
        shap_df = shap_df.round(4)
        logger.info(f"shap value calculated and stored in dataframe")

        logger.info(f"Melting the shap dataframe")
        melted_shap_df = pd.melt(shap_df, id_vars=["CLAIMANT_ID", "CLAIM_APPLICATION_ID"])
        melted_shap_df = melted_shap_df.rename(columns={"variable":"FEATURE_COLS", "value":"VALUE"})
        melted_shap_df_path = os.path.join(outputpath, inputdata["model_feature_importance"], "model_explainability_shap_values.csv")
        csv_buffer = StringIO()
        melted_shap_df.to_csv(csv_buffer, line_terminator="\n", index=False)
        with file_system.open(melted_shap_df_path, mode="wt", encoding="utf-8") as f:
            f.write(csv_buffer.getvalue())
        logger.info(f"shap calculated values saved at {melted_shap_df_path}")
        
    except Exception as e:
        traceback.print_exc(limit=100, file=sys.stdout)
        logger.error("Exception during prediction using unemployment insurance model!", exc_info=True)
        sys.exit("Exception during prediction using unemployment insurance model!")


# #### Please save the notebook after any change to ensure the correct code extraction.

<<<<<<< HEAD
# In[13]:
=======
# In[20]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


save_notebook()


# #### Please execute the following function everytime the notebook is renamed.

# *********** PLEASE ENSURE TO EXECUTE THE BELOW SHELL BEFORE PROCEEDING FOR MODEL ONBOARDING ************ 
# #### Please execute the following cell and provide the model independent and dependent variable information for creation of metadata file.

<<<<<<< HEAD
# In[14]:
=======
# In[16]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


enter_metadata_information(2)


# ### Onboard Model to CDMS 
# * 			Please execute the following cell to onboard the model to the CDMS server. 
# * 			Before onboarding the model, please ensure that the programmable interface for both training and prediction are present in the notebook and working. 
# * 			Please execution this cell will push a new version of the model to CDMS

<<<<<<< HEAD
# In[15]:
=======
# In[29]:
>>>>>>> 61e9c5cc07419aa1dd723560ab45d51bfb00cd81


metadata_dict['db_interaction'] = False

onboard(metadata_dict, 
updt_db_cust_mdl=False, 
trained_mdl_type=None )


# Below function is provided to onboard model to remote servers. To perform the activity, please provide correct set of information for each servers.

# rs_dict={
#     "server_1":{"host":"", "username":"", "password":"", "remote_path":""},
#     #"server_2":{"host":"", "username":"", "password":"", "remote_path":""}
# }  
# remote_onboarding(rs_dict)
# 
# onboard(metadata_dict, 
# updt_db_cust_mdl=False, 
# trained_mdl_type=None, 
# onboard_to_CDMS=True, 
# no_of_remote_servers_to_onboard=0, 
# version_increment=True)
