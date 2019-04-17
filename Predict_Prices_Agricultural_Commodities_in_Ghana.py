#!/usr/bin/env python
# coding: utf-8

# The datasets used in this session is obtained from (https://data.gov.gh/dataset/agricultural-production-estimates-1993-2017). The national wholesale price of selected commodities is the average of collated prices of each commodity in the regions.
# 
# There are a number of features in the datasets where it is not readily apparent what the encode or what their units are. Common definitions are:
# 
# * Region: regions in Ghana.
# * Year: Years spanning between 2008 and 2012.
# * Total_Rainfall (mm) : rainfall record in milliimeters (daily).
# * Commodity: variety of farm produce.
# * Weight(Kg): weight of produce in 100kg
# 
# The objective of this hands-on practicals is to build a regresssion model to predict prices for selected commodities in Ghana.

# In[1]:


# import relevant libraries to help with loading the data and plotting
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from pylab import rcParams
from collections import Counter
rcParams['figure.figsize'] = 8,8
get_ipython().run_line_magic('matplotlib', 'inline')


# ### LOADING DATA

# In[2]:


rainfall = pd.read_csv('/home/francisanokye/AIMS/Data_Bank/RAINFALL.csv')
temperature = pd.read_csv('/home/francisanokye/AIMS/Data_Bank/Monthly_Average_Temperature_in_Ghana.csv')
prices = pd.read_csv('/home/francisanokye/AIMS/Data_Bank/NATIONAL_WHOLESALE_PRICE_OF_SELECTED_COMMODITIES_1970_2017.csv',encoding='latin-1')


# In[3]:


# checking the total number of rows and columns for the rainfall data, and print out the first 5 rows
print(rainfall.shape)
rainfall.head()


# In[4]:


# checking data information
rainfall.info()


# In[5]:


# renaming columns for rainfall 
col = {'YEAR':'Year','REGION':'Region','TOTAL RAINFALL(MM)':'Total_Rainfall (mm)'}
rainfall.rename(columns=col,inplace=True)
rainfall.head()


# In[6]:


# check the total number of rows and columns for the temperature data, and print out the first 5 rows
print(temperature.shape)
temperature.head()


# In[7]:


# checking data information
temperature.info()


# In[8]:


# remove unwanted whitespaces in the column names
temperature.columns = temperature.columns.str.strip()
temperature.info()


# In[9]:


# checking the total number of rows and columns for the prices data, and print out the first 5 rows
print(prices.shape)
prices.head()


# In[10]:


prices.info()


# In[11]:


# remove unwanted whitespace in the price column
prices.columns = prices.columns.str.strip()
prices.info()


# In[12]:


# Rename the prices columns
"""
prices.columns = prices.columns.lower()....this does same work as below
"""
K = {'YEAR':'Year', 
     'MONTH':'Month', 
     'COMMODITY':'Commodity', 
     'WEIGHT, KG PER BAG':'Weight(Kg)',
     "PRICE, ¢ GH":"Prices(GH¢)"
    }
prices.rename(columns=K,inplace=True)
prices.head()


# In[13]:


# The month column for the prices data are in strings so i would like to convert them into numbers 
# to match that in the temperature data
"""
prices['Month'] = prices['Month'].replace({'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,'JULY':7,
             'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12})
"""
M = {'NOVEMBER': 11,
     'SEPTEMBER': 9, 
     'OCTOBER':10,
     'AUGUST':8, 
     'APRIL':4,  
     'MARCH':3 ,
     'MAY':5,        
     'DECEMBER':12,
     'JULY':7,  
     'FEBRUARY':2, 
     'JANUARY':1 , 
     'JUNE':6
    }
prices['Month'] = prices['Month'].map(M)


# In[14]:


# checking out on the changes made so far on prices
prices.head()


# In[77]:


# create a date column and set as index to merge the data on
# prices['Date'] = pd.to_datetime(prices[['Year','Month']].assign(Day=1))
# prices = prices.set_index('Date')
# prices.head()


# ### MERGING DATA

# In[15]:


# merging the rainfall and temperature data on the year column where the data intersect, and we call it df
df = pd.merge(rainfall,temperature,on='Year',how='inner')
print(df.shape)
df.head()


# In[16]:


# now we merge the df data just created from rainfall and temoerature with prices on both year and month
DF = pd.merge(df,prices,on=['Year','Month'],how='inner')
print(DF.shape)
DF.head()


# In[17]:


# check for missing or null values in the merged data called DF
DF.isnull().sum()


# In[18]:


# information about our data
DF.info()


# In[19]:


# the data under consideration is fron a single country, ie Ghana
DF.Country.unique()


# In[20]:


# data is gathered from 10 different regions in Ghana
DF.Region.unique()


# In[21]:


Counter(DF['Region'])


# In[22]:


# the merged data spans only period from 2008 - 2012
DF.Year.unique()


# In[23]:


Counter(DF['Year'])


# In[24]:


# descriptive statistics of the merged data
DF.describe()


# In[26]:


# a plot to visualize the entire data and relate the individual variables with one another setting hue to commodity
sb.pairplot(DF,hue='Commodity')


# In[27]:


# Since the country is just Ghana, it does not offer any more information, hence, we drop it
DF.drop(columns='Country',axis=1,inplace=True)
DF.head()


# In[91]:


# we plot to visualize the numeric columns
C = DF.columns
for k in C:
    if DF[k].dtype != 'object':
        print(k)
        sb.distplot(DF[k],hist=True)
        plt.xlabel(k)
        plt.show()
    


# In[92]:


# we plot to visualize the categorical columns
for d in C:
    if DF[d].dtype == 'object':
        print(d)
        sb.countplot(DF[d])
        plt.show()


# In[93]:


# the commodity variable consists of six different varieties
Counter(DF.Commodity)


# In[94]:


DF['Weight(Kg)'].max()


# In[95]:


DF['Weight(Kg)'].min()


# In[96]:


DF['Weight(Kg)'].mean()


# In[28]:


# correlation plot to visualize how the variables relate with one another
cor = DF.corr()
sb.heatmap(cor,xticklabels= cor.columns.values,yticklabels=cor.columns.values,annot=True,annot_kws={'size':10})
plt.show()


# In[29]:


cor


# In[30]:


X = DF.drop(columns='Prices(GH¢)',axis=1) # drop the target variable 
print(X.shape)
y = DF['Prices(GH¢)'] # drop the other variables and leave out the target variable which is prices
print(y.shape)


# In[31]:


# encoding the columns with sublevels into numeric using scikit-learn labelencoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Converting categorical columns into numeric
X['Commodity'] = le.fit(X['Commodity']).transform(X['Commodity'])
X['Region']    = le.fit(X['Region']).transform(X['Region'])

# year column treated as numeric but in fact it's a class
X['Year']      = le.fit(X['Year']).transform(X['Year']) 
X.shape


# In[32]:


# importing relevant libraries from scikit-learn
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures

# spliting data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[33]:


# normalizing data using the standard scaler since the different variables have different scales of measurement
columns_to_scale = X.columns.tolist()
columns_to_scale = [x for x in columns_to_scale if x != 'Prices(GH¢)']
# print(columns_to_scale)

std_scaler = StandardScaler().fit(X_train[columns_to_scale])
X_train[columns_to_scale] = std_scaler.transform(X_train[columns_to_scale])
X_test[columns_to_scale] = std_scaler.transform(X_test[columns_to_scale])


# ### BUILDING LINEAR  MODELS WITH CROSS VALIDATION

# In[37]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import time


# I will now explore a number of regression algorithms. Their performance when compared to each other may give a first indication on what the most promising algorithms for the dataset are. I can then focus on these to furtehr optimize the hyperparameters.
# 
# For now, I include the following algorithms:
# 
# * linear regression (<a href="http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares">Sklearn</a>)
# * linear regression with L2 regularization (<a href="http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression">Sklearn</a>)
# * polynomial regression (<a href="http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions">Sklearn</a>)
# * random forest regression (<a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor">Sklearn</a>)
# * gradient boosted tree regression (<a href="http://scikit-learn.org/stable/modules/ensemble.html#regression">Sklearn</a>)
# * nearest neighbor regression (<a href="http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression">Sklearn</a>)
# * support vector regression (<a href="http://scikit-learn.org/stable/modules/svm.html#regression">Sklearn</a>)
# 
# All algorithm are first run using their default values.

# #### Linear Regression

# In[38]:


lin = LinearRegression()
before = time.time()
linscores = cross_val_score(lin,X_train,y_train,cv=10,scoring='r2')
print("The training accuracy of linear regression is %0.2f  (+/- %0.2f)" % (linscores.mean(), linscores.std() * 2))
after = time.time()
print('Time taken to execute algorithm: {:5.2f} s'.format(after - before))


# #### Polynomial Regression

# In[39]:


before = time.time()
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
linpoly = LinearRegression()
linpolyscores = cross_val_score(linpoly,X_train_poly,y_train,cv=10,scoring='r2')
print("The training accuracy of polynomial linear regression is %0.2f  (+/- %0.2f)" %(linpolyscores.mean(), linpolyscores.std()*2))
after = time.time()
print('Time taken to execute algorithm: {:5.2f} s'.format(after - before))


# #### GradientBoostingRegressor

# In[40]:


GBR = GradientBoostingRegressor()
before = time.time()
gbrscores = cross_val_score(GBR,X_train,y_train,cv=10,scoring='r2')
print("The training accuracy of gradientboostingregressor is %0.2f  (+/- %0.2f)" %(gbrscores.mean(), gbrscores.std()*2))
after = time.time()
print('Time taken to execute algorithm: {:5.2f} s'.format(after - before))


# #### Ridge Regression (Linear least squares with l2 regularization)

# In[41]:


ridge = Ridge(alpha=.8)
before = time.time()
ridgescores = cross_val_score(ridge,X_train,y_train,cv=10,scoring='r2')
print("The accuracy for the ridge regrssion is %.2f (+/- %.2f)" %(ridgescores.mean(),ridgescores.std()*2))
after = time.time()
print('Time taken to execute algorithm is: {:5.2f} s'.format(after - before))                  


# #### RandomForestRegressor

# In[42]:


randforreg = RandomForestRegressor()
before = time.time()
ranrgscores = cross_val_score(randforreg,X_train,y_train,cv=10,scoring='r2')
print("The accuracy for the random forest regressor is %.2f (+/- %.2f)" % (ranrgscores.mean(),ranrgscores.std()*2))
after = time.time()
print("Time taken to execute algorithm is : {:5.2f} s".format(after - before))


# #### Support vector regressor (SVR)

# In[52]:


kernel = ['linear', 'poly', 'rbf']
for i in kernel:
    print("when Kernel is {}".format(i))
    svr = SVR(kernel=i,C=1)
    before = time.time()
    svrscores = cross_val_score(svr, X_train,y_train,cv=10,scoring='r2')
    print("The accuracy for the support vector regressor is %.2f (+/- %.2f)" % (svrscores.mean(),svrscores.std()*2))
    after = time.time()
    print("Time taken to execute algorithm is {:5.2f}".format(after - before))
    print("***" * 50)


# ## KNeighborsRegressor

# In[53]:


knr = KNeighborsRegressor(n_neighbors=5,n_jobs=-1,weights='uniform')
before = time.time()
knrscores = cross_val_score(knr,X_train,y_train,cv=10,scoring='r2')
print("The accuracy for the nearest neighbor regressor is %.2f (+/- %.2f)" %(knrscores.mean(),knrscores.std()*2))
after = time.time()
print("Time taken to execute algorithm is {:5.2f}".format(after - before))


# The **RandomForestRegressor** has performed the best with perfect accuracy followed by GradientBoostingRegressor and KNeighborsRegressor,

# The GradientBoostingRegressor has been the recent favorite among many machine-learning competitions - it performed relatively better compared to other algorithms except for RandomForestRegressor. Obviously, I did not perform any parameter tuning, but rather used the default values. Now let's see if I can increase the performance by tuning some of the main parameters:

# In[132]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
before = time.time()
tuned_parameters = [{'n_estimators':[100,200,300,400,500],
                     'max_depth':[3,5,7,9,11],
                     'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}]
gradtuned = GradientBoostingRegressor(loss='ls',random_state=0)
estgrad = GridSearchCV(gradtuned,tuned_parameters,cv=10,n_jobs=-1)
estgrad.fit(X_train,y_train)

means = estgrad.cv_results_['mean_test_score']
stds = estgrad.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, estgrad.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
    
after = time.time()
print('Exec. time: {:5.2f} s'.format(after-before))


# Upon the tuning of the hyperparameters, it is obvious that the gradientboosting regressor has seen tremendous improvement.

# In[74]:


GBRfinal = GradientBoostingRegressor(learning_rate=0.1,max_depth=5,n_estimators=200)
before = time.time()
GBRscores = cross_val_score(GBRfinal,X_train,y_train,cv=10,scoring='r2')
print("The training accuracy of gradientboostingregressor is %0.2f  (+/- %0.2f)" %(GBRscores.mean(), GBRscores.std()*2))
after = time.time()
print('Time taken to execute algorithm: {:5.2f} s'.format(after - before))


# It is clearly shown from our built models that GradientBoostingRegressor and RandomForestRegressor have both produced r2_scores of 1 (coefficient of determination), which implies that these models best explain the variability in the data.

# In[ ]:




