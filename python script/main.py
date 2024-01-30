#!/usr/bin/env python
# coding: utf-8

# <font size="6"> <div style="text-align: center"> **Multi Classification predictor on Payment's Timeliness**

# The objective of this task is to build a predictor that informs whether a payment occurs ahead of time, on time or if it's delayed, given an input data containing transactional and master data.
# 
# A payment is considered to be on time if it occurs within the same month of the due date.
# 
# To tackle this problem we will do the following steps:
# 
# - Data Exploration and Analysis
# - Feature  and Data Transformation
# - Forecasting models using ML classification algoritms - Supervised methods to build a predictor

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports</a></span><ul class="toc-item"><li><span><a href="#Data-Upload" data-toc-modified-id="Data-Upload-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Data Upload</a></span></li></ul></li><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span><ul class="toc-item"><li><span><a href="#First-Look" data-toc-modified-id="First-Look-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>First Look</a></span></li><li><span><a href="#Some-Variable's-Analysis" data-toc-modified-id="Some-Variable's-Analysis-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Some Variable's Analysis</a></span><ul class="toc-item"><li><span><a href="#Company,-Country,-City-and-profit-Center" data-toc-modified-id="Company,-Country,-City-and-profit-Center-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Company, Country, City and profit Center</a></span></li><li><span><a href="#Document-Type" data-toc-modified-id="Document-Type-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Document Type</a></span></li><li><span><a href="#Credit-Risk" data-toc-modified-id="Credit-Risk-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>Credit Risk</a></span></li></ul></li><li><span><a href="#Further-variable-transformations" data-toc-modified-id="Further-variable-transformations-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Further variable transformations</a></span><ul class="toc-item"><li><span><a href="#Dependent-Variable-Creation" data-toc-modified-id="Dependent-Variable-Creation-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Dependent Variable Creation</a></span></li><li><span><a href="#DUE_DATE_SOURCE-and-MANSP" data-toc-modified-id="DUE_DATE_SOURCE-and-MANSP-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span><code>DUE_DATE_SOURCE</code> and <code>MANSP</code></a></span></li><li><span><a href="#One-Hot-Encoding---BLART-and-TBSLT_LTEXT" data-toc-modified-id="One-Hot-Encoding---BLART-and-TBSLT_LTEXT-2.3.3"><span class="toc-item-num">2.3.3&nbsp;&nbsp;</span>One Hot Encoding - <code>BLART</code> and <code>TBSLT_LTEXT</code></a></span></li></ul></li><li><span><a href="#Data-Cleaning" data-toc-modified-id="Data-Cleaning-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Data Cleaning</a></span><ul class="toc-item"><li><span><a href="#Outliers" data-toc-modified-id="Outliers-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>Outliers</a></span></li><li><span><a href="#Multicollinearity-Check" data-toc-modified-id="Multicollinearity-Check-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>Multicollinearity Check</a></span></li><li><span><a href="#Recursive-Feature-Elimination" data-toc-modified-id="Recursive-Feature-Elimination-2.4.3"><span class="toc-item-num">2.4.3&nbsp;&nbsp;</span>Recursive Feature Elimination</a></span></li></ul></li></ul></li><li><span><a href="#Data-Transformation" data-toc-modified-id="Data-Transformation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Transformation</a></span><ul class="toc-item"><li><span><a href="#Normalization" data-toc-modified-id="Normalization-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Normalization</a></span></li></ul></li><li><span><a href="#Modelling" data-toc-modified-id="Modelling-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Modelling</a></span><ul class="toc-item"><li><span><a href="#Model-Evaluation" data-toc-modified-id="Model-Evaluation-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Model Evaluation</a></span></li><li><span><a href="#Simple-Models" data-toc-modified-id="Simple-Models-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Simple Models</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>Logistic Regression</a></span></li><li><span><a href="#Neural-Networks" data-toc-modified-id="Neural-Networks-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>Neural Networks</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-4.2.3"><span class="toc-item-num">4.2.3&nbsp;&nbsp;</span>Random Forest</a></span></li></ul></li><li><span><a href="#Tuned-Model" data-toc-modified-id="Tuned-Model-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Tuned Model</a></span><ul class="toc-item"><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>Random Forest</a></span></li></ul></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

# # Imports

# In[288]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import itertools
import random

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot 

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error, plot_confusion_matrix, plot_roc_curve, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Upload

# In[2]:


df = pd.read_csv('training_data.csv')
df


# # Exploratory Data Analysis

# ## First Look

# In[3]:


# To describe dataset structure - nº of observations(rows) and features(columns)
df.shape


# In[4]:


# To identify dataset features (columns)
df.columns


# In[5]:


# To check dtypes and missing values
df.info()


# <font size="3">All but **4 columns** have null records: `MANSP`, `CTLPC`, `HISTORICRATING` and `CURRENTRATING`.

# In[6]:


# To describe the main statistical informartion about the numerical features

df.describe().T


# In[7]:


# To describe object statistical information on all columns

df.astype(object).describe().T


# <font size="3"> 
# - `MANDT` and `MWST2` (client code and tax information) have only one value each variable - irrelevant and so it will be deleted
# - `GJAHR` and `GJAHR2` correspond to fiscal year information on each document, however it is not relevant for the purpose of this problem - it will be deleted
# - `DATUM` and `DOCUMENT_DATE` represent the date in which the document was issued (the first one only the date and the second one in unix timestamp format) and both are irrelevant since we only want to see differences between the payment date (`PAYMENT_DATE`) and the date when payment has to be made (`DUE_DATE`) - it will be deleted   
# 

# ## Some Variable's Analysis

# <font size="3">Now, let's compare interesting relationships between some groups of variables.

# ### Company, Country, City and profit Center

# <font size="3">First, let's check for the relationship between **4 variables** about the location and specifics of the payable company: `KUNNR`, `PRCTR`, `KNA1_LAND1` and `KNA1_ORT01`.

# In[8]:


#df.groupby(['KUNNR', 'KNA1_LAND1', 'KNA1_ORT01']).count().reset_index().nunique()
a = df.groupby('KUNNR')['KNA1_LAND1','KNA1_ORT01'].agg(['unique'])
a


# In[9]:


a.astype(object).describe().T


# <font size="3">The above indicates us that every `KUNNR` corresponds to one country and one city only because this group by contains 23067 unique values, which is equal to those unique values, as seen previously. This means, that for each customer information, there's only one combination in country and city, and, thus, it facilitates our job onwards, as both categorical fields (country and city) can be removed, while the center code is kept.

# In[10]:


b = df.groupby('KUNNR')['PRCTR'].agg(['unique'])
b


# In[11]:


b.astype(object).describe().T


# <font size="3">Above, it is visible that each company code `KUNNR` can have more than one profit center (`PRCTR`) as there are 5688 unique values grouped by the first column but only 226 unique profit centers, meanng that different countries and cities can have the same profit center and this information can be useful for the predictive model, therefore, I will keep this column as well.

# ### Document Type

# <font size="3"> Other interesting comparison can be between three categorical variables relative to each record's document/operation type: `BLART`, `T003T_LTEXT` and `TBSLT_LTEXT`

# In[12]:


# relation between 'BLART' and 'T003T_LTEXT'
sns.countplot(x='BLART', hue='T003T_LTEXT', data=df)


# In[13]:


# percentage of unique values in 'BLART'
df['BLART'].value_counts()/df['BLART'].count()


# <font size="3"> 
# - From the first countplot, we see that each value from `BLART` corresponds to a unique value from `T003T_LTEXT`, therefore they have the exact same information and one can be removed.
# - Secondly, we see an extreme dominance in the **'RV'** type or **'Billing doc transfer'** with about 98.4% of total records. Further ahead I will check if it still makes sense to keep this column (depending on the impact onto the output variable). 
# 

# In[14]:


# percentage of unique values in 'TBSLT_LTEXT' and unique count

print(df['TBSLT_LTEXT'].value_counts()/df['TBSLT_LTEXT'].count())
df['TBSLT_LTEXT'].value_counts()


# <font size="3">With `TBSLT_LTEXT` we see again a very dominant value which is 'Invoice' as a type of payment with about 93% of total records. Ahead, I will check again if it worth to keep this column or how to transform it.

# ### Credit Risk

# <font size="3"> 
# - Here, we'll compare `CTLPC`, `HISTORICRATING` and `CURRENTRATING`. They all seem related to credit risk ratings, being the first one more related with levels. First let's transform the **rating** variables into numbers only (by removing the 'R' and the signal, so I will consider a rating R2+ equal to R2- and R2, for instance) and, then, see correlation between them, which should be really high.
# - For null values, let's fill these values as a value **11** to indicate that they are 'Not Rated' or missing.
# 

# In[15]:


#transform 'Historicrating' and 'currentrating'

df['HISTORICRATING'] = df['HISTORICRATING'].str[1]
df['CURRENTRATING'] = df['CURRENTRATING'].str[1]
df


# In[16]:


# Replace nulls with the value 11

df.update(df[['CTLPC', 'HISTORICRATING', 'CURRENTRATING']].fillna(11))
print(df[['CTLPC', 'HISTORICRATING', 'CURRENTRATING']])
df.info()


# In[17]:


# transform these columns into integer

df[['CTLPC', 'HISTORICRATING', 'CURRENTRATING']] = df[['CTLPC', 'HISTORICRATING', 'CURRENTRATING']].astype(int)
df.info()


# In[18]:


# See Pearson's correlation between these 3 columns

def heatmap(df=df):
    corr_df = df.corr(method='pearson')

    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(corr_df, annot=True, square=False, mask=mask)
    return plt.show()

heatmap(df[['CTLPC', 'HISTORICRATING', 'CURRENTRATING']])


# <font size="3"> As seen above and as expected, all variables are extremely well correlated (especially the historical and current rating). Therefore, I will remove `HISTORICRATING` to avoid multicollinearity issues further on the model but still keep `CTLPC` for now.

# In[84]:


# delete all columns so far

df1 = df.drop(columns=['MANDT', 'MWST2', 'GJAHR', 'GJAHR2', 'DATUM', 'DOCUMENT_DATE',                       'KNA1_LAND1', 'KNA1_ORT01', 'T003T_LTEXT', 'HISTORICRATING'])
df1


# ## Further variable transformations

# ### Dependent Variable Creation

# <font size="3"> 
# Finally, let's create the dependent variable which is the ouput for our predictor. <br>
# This is a ternary classification problem because we have three outputs for calculations:
# - A payment done in time considering that it was done on the same month as the due_date (Value **0**)
# - A payment done ahead of time considering that it was done one or more months before the due_date month (Value **1**)
# - A delayed payment considering that it was done one or more months after the due_date month (Value **2**)

# In[85]:


# Convert `PAYMENT_DATE` column from UNIX timestamp to date and `DUE_DATE` from object to date

df1['PAYMENT_DATE'] = pd.to_datetime(df1['PAYMENT_DATE'], format='%Y%m%d')
df1['DUE_DATE'] = pd.to_datetime(df1['DUE_DATE'])

df1[['PAYMENT_DATE', 'DUE_DATE']]


# In[91]:


# create column to see difference in months (payment date - due date)

df1['diff_months'] = (df1['PAYMENT_DATE'].dt.year - df1['DUE_DATE'].dt.year) * 12 +            (df1['PAYMENT_DATE'].dt.month - df1['DUE_DATE'].dt.month)

df1


# In[94]:


# Create `Resp` Variable

df1['RESP'] = np.where(df1['diff_months'] == 0, 0, np.where(df1['diff_months'] >  0, 2, 1)) 

df1['RESP'].value_counts()


# In[109]:


# Visualize in a pie chart
data1 = df1['RESP'].value_counts()
pie, ax = plt.subplots(figsize=[8,8])
labels = data1.keys()

plt.pie(x=data1, autopct=lambda p : '{:.2f}%  ({:,.0f})'.format(p,p * sum(data1)/100), labels=labels)
plt.title('Response Variable Distribution', size=14)

plt.show()


# <font size="3"> We can see a predominence in payments done in time (in the same month as the due date) and, then, more delayed payments than upfront ones, which makes sense and, therefore, we can't consider this variable too much unbalanced, which might be better for the modelling phase. 

# ### `DUE_DATE_SOURCE` and `MANSP`

# <font size="3"> 
# Since `DUE_DATE_SOURCE` and `MANSP` are categorical variables, let's transform them into numerical.
# - For the first one, I will keep only the key number (0 to 3), being 0 = 'ZFBDT'. This information can be useful to tell us what source it was used for due date calculations (with or without cash discounts)
# - For `MANSP`, since it has a lot of missing values and the value 'H' is dominant in valid values, I will consider 0 all nulls (meaning there wasn't any notice to the client) and 1 to all non-blank values (considering that the customer was contacted (dunning process).

# In[110]:


# encode values in `DUE_DATE_SOURCE`

df2 = df1
df2.replace({'DUE_DATE_SOURCE' : { 'ZFBDT' : 0, 'ZBD1T' : 1, 'ZBD2T' : 2, 'ZBD3T': 3 }}, inplace=True)

df2['DUE_DATE_SOURCE']    


# In[111]:


# print number of missing values in `MANSP`

print('`MANSP` variable has %.1f%% missing values.' % (df1['MANSP'].isnull().mean()*100))


# <font size="3"> Although only 2.9% of values are valid, this variable might be useful to predict essentially delayed payments, since its the objective, therefore I will keep using it, encoding all non-null as 1 and null as 0 like explained above.

# In[112]:


# encode values in `MANSP`

df3 = df1
df3['MANSP'] = (df3['MANSP'].notnull()).astype('int')
print(df3['MANSP'])
df3['MANSP'].value_counts()


# In[115]:


# relation between 'MANSP' and 'RESP'
sns.countplot(x='MANSP', hue='RESP', data=df3)


# <font size="3"> This graph actually concludes the hypothesis in our sample that dunning a client (`MANSP` = 1) is more prevalent when the client is going to pay earlier than the due date, than on the same month or even when the payment comes months later.

# ### One Hot Encoding - `BLART` and `TBSLT_LTEXT`

# <font size="3"> Now it is time to transform our two categorical variables into multiple binary using the One Hot Encoding method.

# In[122]:


# relation between 'BLART' and 'RESP' amplified

ax = sns.countplot(x='BLART', hue='RESP', data=df3)
ax.set(ylim=(0, 5000))


# In[123]:


# Transform `BLART` variable

list_bin = [1 if value == 'RV' else 0 for value in eval('df3[' + '\'' + 'BLART' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'RV' + '\'' + '] = list_bin')

list_bin = [1 if value == 'ZA' else 0 for value in eval('df3[' + '\'' + 'BLART' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'ZA' + '\'' + '] = list_bin')

list_bin = [1 if value == 'ZF' else 0 for value in eval('df3[' + '\'' + 'BLART' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'ZF' + '\'' + '] = list_bin')


# In[125]:


# Transform `TBSLT_LTEXT` variable

list_bin = [1 if value == 'Invoice' else 0 for value in eval('df3[' + '\'' + 'TBSLT_LTEXT' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'Invoice' + '\'' + '] = list_bin')

list_bin = [1 if value == 'advances Third Party' else 0 for value in eval('df3[' + '\'' + 'TBSLT_LTEXT' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'advances Third Party' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Credit memo' else 0 for value in eval('df3[' + '\'' + 'TBSLT_LTEXT' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'Credit memo' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Down payment request' else 0 for value in eval('df3[' + '\'' + 'TBSLT_LTEXT' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'Down payment request' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Incoming payment' else 0 for value in eval('df3[' + '\'' + 'TBSLT_LTEXT' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'Incoming payment' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Payment difference' else 0 for value in eval('df3[' + '\'' + 'TBSLT_LTEXT' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'Payment difference' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Reverse credit memo' else 0 for value in eval('df3[' + '\'' + 'TBSLT_LTEXT' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'Reverse credit memo' + '\'' + '] = list_bin')

list_bin = [1 if value == 'Reverse invoice' else 0 for value in eval('df3[' + '\'' + 'TBSLT_LTEXT' + '\'' + ']')]
exec('df3.loc[:, ' + '\'' + 'Reverse invoice' + '\'' + '] = list_bin')

df3


# ## Data Cleaning

# ### Outliers

# <font size="3"> Now let's inspect for outliers in every numerical variable by visualizing each distribution in whisker's boxes:

# In[130]:


# first, select all numeric variables

numerics = ['int32', 'int64', 'float32', 'float64']
dfo = df3.select_dtypes(include=numerics)

dfo.info()


# In[133]:


fig, axes= plt.subplots(figsize=(16,30))
whiskers=dict()
for o in range(len(dfo.columns)):
    plt.subplot(6, 4, o+1)
    C1=plt.boxplot(x = dfo.columns[o], data = dfo)
    plt.title(dfo.columns[o])
    w=[item.get_ydata() for item in C1['whiskers']]
    whiskers[dfo.columns[o]] = (w[0][1],w[1][1])


# <font size="3"> 
#     
# - As expected very few columns have outliers and those only columns with real and interpretable extreme values are `VALUE_EUR` and `MWSTS`: Value to be paid and tax amount.
# - With the analysis of the summary statistics with these variables it is noticeable that there are a lot of 0s and therefore, the median, 25 and 75 percentile are closer to 0 (or they are really low when compared to max and extreme values), which push the standard deviation to really high values.
# - Since the standard deviation is really high, I will delete all records whose values are above (and below if exists) 3 standard deviations comparing to the mean.

# In[149]:


value = df3['VALUE_EUR']

out_value = df3[(value >= (value.mean() + 3*value.std())) | (value < (value.mean() - 3*value.std()))]
out_value


# In[150]:


val = df3['MWSTS']

out_val = df3[(val >= (val.mean() + 3*val.std())) | (val < (val.mean() - 3*val.std()))]
out_val


# In[161]:


# remove outliers

df4 = df3.drop(out_value.index | out_val.index)
df4


# ### Multicollinearity Check

# In[162]:


# check one last time multicollinearity issues with all variables in this case

heatmap(df4)


# <font size="3"> Here, we see an interesting medium correlation between `ZTERM` and the rating variables. Other than that, we see some strong negative correlations between some binary variables created but these don't have a practical meaning and therefore I will keep all columns right until the last dimension reduction exercise (Recursive Feature Elimination).

# <font size="3"> But first, let's remove already useless columns:

# In[166]:


df5 = df4.drop(columns =['BLART', 'TBSLT_LTEXT', 'DUE_DATE', 'PAYMENT_DATE', 'diff_months'])
df5.head(5)


# ### Recursive Feature Elimination

# <font size="3"> For this step I am going to use a Decision Tree classifier as the method for feature importance to choose for the final treated dataset.

# In[183]:


x = df5.drop(columns =['RESP'])
y = df5['RESP']

# for here let's split the traditional way with 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)


# In[184]:


model_gini = DecisionTreeClassifier(criterion = 'gini').fit(x_train, y_train)

def plot_feature_importances(model):
    n_features = x_train.shape[1]
    plt.figure(figsize=(20,10))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_train.columns)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.show()


# In[185]:


plot_feature_importances(model_gini)


# In[186]:


model_entropy = DecisionTreeClassifier(criterion = 'entropy').fit(x_train, y_train)

plot_feature_importances(model_entropy)


# <font size="3"> I decided to test two models for RFE, choosing the 'gini' or 'entropy' criterion for the Decision Tree and the results were very similar in terms of assessing the most important features. I will choose as a fair threshold the value of 0.02 and therefore only the following 8 variables will be used further on and so our model will remove any kind of noise and irrelevant features:<br>
# - `Invoice`, `CURRENTRATING`, `VALUE_EUR`, `ZTERM`, `PRCTR`, `KUNNR`, `HKONT` and `BUKRS`.

# In[176]:


df_treated = df5[['Invoice', 'CURRENTRATING', 'VALUE_EUR', 'ZTERM', 'PRCTR', 'KUNNR', 'HKONT', 'BUKRS', 'RESP']]
df_treated


# <font size="3"> 
# 
# - Now, that every variable was treated and analysed, the following table provides a summary of every column description and decision made:

# | Column          	| Description                                      	| Data Info and Importance                                                           	| Action                   	|
# |-----------------	|--------------------------------------------------	|------------------------------------------------------------------------------------	|--------------------------	|
# | MANDT           	| Client                                           	| Always the same value                                                              	| Remove                   	|
# | BUKRS           	| Company Code                                     	| Numerical values with low/medium   importance                                      	| Keep                     	|
# | GJAHR           	| Fiscal Year                                      	| Only two year values - not relevant   for this purpose                             	| Remove                   	|
# | HKONT           	| General Ledger                                   	| 14 values with some importance                                                     	| Keep                     	|
# | KUNNR           	| Customer information                             	| Several numerical values and has   really good importance                          	| Keep                     	|
# | PRCTR           	| Profit Center                                    	| Several numerical values and has good   importance                                 	| Remove                   	|
# | KNA1_LAND1      	| Country code                                     	| Categorical values, but related with   'PRCTR'                                     	| Remove                   	|
# | KNA1_ORT01      	| City code                                        	| Categorical values, but related with   'PRCTR'                                     	| Keep                     	|
# | ZTERM           	| Term of payment Key                              	| Several numerical values and has good   importance                                 	| Keep                     	|
# | DUE_DATE_SOURCE 	| Due Date Calculation term keys                   	| 4 Categorical values without   importance                                          	| Remove                   	|
# | VALUE_EUR       	| Payment in euros                                 	| Several numerical values and it's the   most important variable with some outliers 	| Keep                     	|
# | MWSTS           	| Tax   Amount in Local Currency                   	| Several numerical values with some   outliers, however, it's not important         	| Remove                   	|
# | MWST2           	| LC2   tax amount information                     	| Always the same value                                                              	| Remove                   	|
# | BLART           	| Document type code                               	| 3 Categorical values without   importance                                          	| Remove                   	|
# | T003T_LTEXT     	| Document type description                        	| 3 Categorical values which correspond   to same values in 'BLART'                  	| Remove                   	|
# | TBSLT_LTEXT     	| Payment type operation                           	| 8 Categorical values in which it only   matter if it's an 'Invoice' or   not       	| Remove all but 'Invoice' 	|
# | MANSP           	| Dunning   lock reason information                	| 97 % blanks and overall has low   importance                                       	| Remove                   	|
# | CTLPC           	| Credit risk levels                               	| 11 values and several blanks with low   importance                                 	| Remove                   	|
# | HISTORICRATING  	| Historical credit rating                         	| Alphanumerical column with almost   perfect correlation with 'CURRENTRATING'       	| Remove                   	|
# | CURRENTRATING   	| Current credit rating                            	| Alphanumerical column with medium   importance but with several blanks             	| Keep                     	|
# | DATUM           	| Date of document emission                        	| Date column irrelevant for this   problem                                          	| Remove                   	|
# | DUE_DATE        	| Date where invoice has to be paid                	| Date column only relevant for the   predictor variable creation                    	| Remove                   	|
# | GJAHR2          	| Fiscal Year 2                                    	| 6 year values - not relevant for this   purpose                                    	| Remove                   	|
# | DOCUMENT_DATE   	| Date of document emission (timestamp   included) 	| Timestamp column irrelevant for this   problem                                     	| Remove                   	|
# | PAYMENT_DATE    	| Date when payment was done                       	| Timestamp column only relevant for   the predictor variable creation               	| Remove                   	|
# |                 	|                                                  	|                                                                                    	|                          	|

# # Data Transformation

# ## Normalization

# <font size="3"> 
#     
# - Now, the final step for the modelling phase is to normalize our dataset, since variables like `VALUE_EUR` and `KUNNR` have really higher unit dimensions when compared to remaining features. <br>
# - The method for keeping all variables at the same level that I choose is the min-max normalization (0 to 1) which will keep every column with values ranging from 0 to 1:

# In[308]:


# include all but the Response variable
df6 = df_treated.iloc[:,:-1]

# Create an instance of the MinMaxScaler class
min_max_scaler = MinMaxScaler()

# Normalize our dataset (returns an array)
normalized_df = min_max_scaler.fit_transform(df6)

# Convert the array above back into a pandas DataFrame
normalized_df = pd.DataFrame(data=normalized_df, columns=df6.columns, index=df6.index)

#include `RESP` now
normalized_df['RESP'] = df_treated['RESP']
normalized_df


# <font size="3"> <ins> Notice that now the target variable is the only one which is not normalized, however it would be the same to do so, it's just for same interpretability reasons. </ins>

# # Modelling

# <font size="3"> 
# 
# - Now, it comes the modelling phase, where I'm going to compute and compare 3 simple models: 1 **Logistic Regression**, 1 **Random Forest** and 1 **Neural Network** and, therefore, fine-tune the best model in order to try to achieve the best possible in terms of accuracy and other metrics.

# In[270]:


# separate X and Y variables
y = normalized_df['RESP']
x = normalized_df.iloc[:,:-1]

# again, split the same way
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)


# ## Model Evaluation

# <font size="3"> I'll begin with the model evaluation functions in order to compare model results right away.

# In[273]:


# Classification reports and confusion matrix 

def model_eval(model):
    m = model.fit(X_train, y_train)
    pred = m.predict(X_test)
    print('\nClassification Report for Train:\n', classification_report(y_train, model.predict(X_train)))
    print('\nClassification Report for Test:\n',classification_report(y_test, model.predict(X_test)))
    plot_confusion_matrix(model, X_test, y_test)
    plt.title('Confusion Matrix for Test')
    plt.show()


# ## Simple Models

# ### Logistic Regression

# In[274]:


# model definition 
model_lr_simple = LogisticRegression(random_state=0)    


# In[275]:


model_eval(model_lr_simple)


# <font size="3"> Similar results for Train and Test, meaning that the model does not shows any signs of overfitting, however, the total accuracy and values for F1 score (especially when trying to predict the least frequent classes) are not the most satisfying and, thus, this is not probably the best model for this context.

# ### Neural Networks

# In[278]:


# model definition 
model_nn_simple = MLPClassifier(random_state=0)    


# In[279]:


model_eval(model_nn_simple)


# <font size="3"> This simple NN seems to be an improvement to the simple LR and still doesn't show signs of overfitting.

# ### Random Forest

# In[276]:


# model definition 
model_rf_simple = RandomForestClassifier(random_state=0)    


# In[277]:


model_eval(model_rf_simple)


# <font size="3"> Here, model improves, however, there is a sign of overfitting as the train results are really good but the test ones show us just medium to good results.<br>
# However, I will keep using this model and, in the next section, fine tune it in order to get the most out of it in terms of constructing a good predictor.

# ## Tuned Model

# <font size="3"> To finalize and to improve the RF model, I will fine tune its most relevant hyperparameters using the RandomizedSearchCV, as it is faster in computation time and has overall better performance in lower dimensionality problems (like this one), when compared with other search methods like GridSearchCV, for instance. 

# ### Random Forest

# In[294]:


# Using RandomizedSearchCV 

rf_params = {
    'n_estimators': [10, 30, 100],
    'max_depth': [2, 6, 10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [3, 6]
}


# In[295]:


best_rf = RandomizedSearchCV(model_rf_simple, rf_params)

best_rf.fit(X_train, y_train)

print(f"Optimal Parameters: {best_rf.best_params_}")


# In[296]:


# model definition 

best_rf = RandomForestClassifier(n_estimators=100 , max_depth=15 , min_samples_split=5, min_samples_leaf=6                                 , random_state=0)    


# In[297]:


model_eval(best_rf)


# <font size="3"> Finally, the model has seen a nice improvement, especially because it is not overfitted anymore (train and test scores look similar) and the overall accuracy achieved with these variables in the model is about **78%**. Still, it lacks in terms of other metrics like 'Recall' or 'F1 Score' especially in less frequent labels.

# # Conclusion

# <font size="4"> 
# 
# - The purpose of this work was, essentially, to try to understand the variables of the dataset and know how to pre-process them, since that the majority of work in machine learning comes from data scrapping and cleaning. <br><br>
# - Our best performing model in this dummy example in building a good predictor was the **Random Forest Classifier**. <br><br>
# - Although the estimator yields good results on payments done on time, we observed that it is the most difficult to predict whenever payments are delayed as the obtained metrics were not favourable. <br><br>
# - Further modelling and comparison with other type of models and parameter tuning should be made as next steps, in order to improve, essentially, **delayed and early payment** predictions.<br><br>
# - Model deployment should be done, finally, as last step to start using the model! 
