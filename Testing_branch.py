#!/usr/bin/env python
# coding: utf-8

# <div style='font-size:30px; color:green; font-weight: bold; margin-bottom: 20px'>Model Builder Template</div> 
#  			<div style='font-size:25px; font-weight: bold'>Overview</div>
# 
# * 			This template is used for seamless integration with CDMS serving layer by preparing `custom_model.py` and `metadata.json` files.
# * 			When this template is opened, a folder with name same as Model Code was created in the local. 
# * 			The cells with code containing <i>import statements</i> or <i>function definitions</i> will be extracted to the `custom_model.py`.
# * 			Please refer to the Model Onboarding User Guide to get the detailed working of this template.

# <div style='font-size:20px'>Note: Rename your notebook file to some meaningful model before doing 			any model development. Changing notebook name after onboarding can make your code behave erroneously.</div>

# ## 1. Model Informatoin

# ### 1.1 Initialize metadata

# In[ ]:


initialize_metadata()


# In[ ]:


#test


# In[1]:


get_ipython().system('git status')


# In[4]:


get_ipython().system('git stash branch demo_iris')


# In[ ]:


get_ipython().system('git add ')


# ### 1.2 Meta data Information
# * 			Please enter a unique model code, model name (e.g. Customer Churn, Propensity to Default, CLTV etc) and Model description.
# * 			Please ensure the 'Create Workspace' button is clicked.
# * 			For more information, please refer to the User Guide.

# In[ ]:


enter_metadata_information(1)


# ## 2. Import Packages
# * 			The following cell can be used to import all the required python packages.
# * 			Please import the `gr` package to use the pre-defined models and functionalities. 			For more information, please refer to the User Guide.

# In[ ]:


#imports 
import os, sys, pickle;


# ## 3. Read Data from Source
# * 			The following code snippet can be used to read the data from different types of sources.
# * 			Please provide the correct details and appropriate queries to read the data. **Note**: This is optional.
# * 			The data can also be read using the functionality provided by the `gr` package.
# * 			To read from s3 use: df = read_data('s3://file Path')

# In[ ]:


df=get_data_from_db(
dbtype='postgresql',
username='retailuser',
password='XXXXXXX',
host='172.18.128.14',
port='5533',
database_name='APT_POC',
sql_query='select * from <<table>>'
)
df.shape


# In[ ]:


def read_data(inpath):
    return pd.read_csv(inpath)


# ## 4. Exploratory Data Analysis 
# * 			The following code snippet provided can be used to get first hand information on the dataset that is going to be analysed. 
# * 			After fetching details to `pandas.dataframe df`, please execute the shell to get the data analysis report. 
# * 			<div style='font-size:15px; font-weight: bold'>Warning** : This is not recommended for a huge dataset as it may take a longer time to generate the report.</div>

# In[ ]:


exploratory_data_analysis(df)


# ## 5. Preprocessing

# In[ ]:





# ## 6. Feature Selection

# In[ ]:





# <div style='font-size:20px; font-weight:bold'>Save Model</div> 
# 
# * 			The following method can be used to save the trained model in the model folder structure created. 
# * 			The method parameter `trained_model` accepts the estimator(trained model) that is to be stored as model `pickle` file. 
#  			The method parameter `filename` accepts a `string` containing the pickle file name.

# In[ ]:


def save_model(trained_model, file_name):
    model_pickle_path = None
    try:
        model_pickle_path = os.path.abspath(os.path.join(__file__, os.pardir, file_name))
    except Exception:
        model_pickle_path = os.path.join(os.path.abspath(''),file_name)
    with open(model_pickle_path, 'wb') as pickle_out:
        pickle.dump(trained_model, pickle_out)


# <div style='font-size:20px; font-weight:bold'>Load Model</div> 
# 
# * 			The following method can be used to load the saved model from the model folder structure. 
# * 			The method parameter `filename` accepts a `string` containing the model pickle file name that is to be loaded.

# In[ ]:


def load_model(file_name):
    model = None
    model_pickle_path = None
    try:
        model_pickle_path = os.path.abspath(os.path.join(__file__, os.pardir, file_name))
    except Exception:
        model_pickle_path = os.path.join(os.path.abspath(''),file_name)
    with open(model_pickle_path, 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    return model


# ## 7. Model Training 
# * 			The methods `train_model(inputdata:dict)` and `get_prediction(inputdata:dict)` are mandatory and needs to be present in the `custom.py` file. 
# * 			The parameter `inputdata` in both `train_model()` and `get_prediction()` needs to be a dictionary. 
# 
#  			For Example: 
#  `inputdata= { "outputpath" : "**/output/",  "inputfilepath" : "filepath" }`

# In[ ]:


def train_model(inputdata: dict):
    '''This method is used for building and training the Model.'''
    pass


# ## 8. Model Evaluation

# In[ ]:





# ## 9. Model explainability

# In[ ]:





# ## 10. Model Scoring

# In[ ]:


def get_prediction(inputdata: dict):
    '''This method is used for predicting the output when appropriate data is fed to the trained model.'''
    pass


# ## 11. Model Onboarding

# <div style='font-size:20px; font-weight:bold'> Please save the notebook after any change to ensure the correct code extraction.</div>

# In[ ]:


save_notebook()


# <div style='font-size:15px; font-weight:bold'> Please execute the following function everytime the notebook is renamed.</div>

# *********** PLEASE ENSURE TO EXECUTE THE BELOW SHELL BEFORE PROCEEDING FOR MODEL ONBOARDING ************ 
#  			<div style='font-size:15px; font-weight:bold'> Please execute the following cell and provide the model independent and dependent variable information for creation of metadata file.</div>

# In[ ]:


enter_metadata_information(2)


# ### 10.1 Push Model to CDMS  
# * 			Please execute the following cell to onboard the model to the CDMS server. 
# * 			Before onboarding the model, please ensure that the programmable interface for both training and prediction are present in the notebook and working. 
# * 			Please execution this cell will push a new version of the model to CDMS

# In[ ]:


metadata_dict['db_interaction'] = False
onboard(metadata_dict, 
updt_db_cust_mdl=False, 
trained_mdl_type=None )


# In[ ]:




