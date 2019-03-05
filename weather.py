#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
import seaborn as sb
from pylab import rcParams
import datetime
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']=7,7


# In[2]:


weather = pd.read_csv("/home/francisanokye/AIMS/Data_Bank/weather.csv")
weather.head()


# In[3]:


weather.shape


# In[4]:


weather.dtypes


# In[5]:


weather.describe()


# In[6]:


weather.isnull().sum()


# ### Replacing missing value for Sunshine

# In[7]:


weather['Sunshine'].isnull().sum()


# In[8]:


missingsunshine = weather[weather['Sunshine'].isnull()].index.tolist()
missingsunshine


# In[9]:


weather['Sunshine'].iloc[missingsunshine] = weather['Sunshine'].mean()


# ### Replacing missing value for WindGustDir

# In[10]:


Counter(weather['WindGustDir'])


# In[11]:


missingwindgustdir = weather[weather['WindGustDir'].isnull()].index.tolist()
missingwindgustdir


# In[12]:


weather['WindGustDir'].iloc[missingwindgustdir] = 'NW'


# In[13]:


sb.countplot(weather['WindGustDir'])


# ### Replacing missing value for WindGustSpeed

# In[14]:


missinggwindGustSpeed = weather[weather['WindGustSpeed'].isnull()].index.tolist()
missinggwindGustSpeed 


# In[15]:


weather['WindGustSpeed'].iloc[missinggwindGustSpeed] = weather['WindGustSpeed'].mean()


# ### Replacing missing value for WindDir9am

# In[16]:


Counter(weather['WindDir9am'])


# In[17]:


sb.countplot(weather['WindDir9am'])


# In[18]:


weather['WindDir9am'].mode()


# In[19]:


pd.crosstab(weather['WindDir9am'].isnull(),weather['WindSpeed9am'].isnull()).transpose()


# In[20]:


missingWindDir9am = weather[weather['WindDir9am'].isnull()].index.tolist()
#missingWindDir9am


# In[21]:


weather['WindDir9am'].iloc[missingWindDir9am] = 'SE'


# In[22]:


sb.countplot(weather['WindDir9am'])


# ### Replacing missing value for WindDir3pm 

# In[23]:


Counter(weather['WindDir3pm'])


# In[24]:


missingWindDir3pm = weather[weather['WindDir3pm'].isnull()].index.tolist()
missingWindDir3pm


# In[25]:


weather['WindDir3pm'].iloc[missingWindDir3pm] = 'NW'


# ### Replacing missing value for WindSpeed9am 

# In[26]:


sb.countplot(weather['WindSpeed9am'])


# In[27]:


missingwindspeed9am = weather[weather['WindSpeed9am'].isnull()].index.tolist()
missingwindspeed9am


# In[28]:


weather['WindSpeed9am'].iloc[missingwindspeed9am] = 6.0


# In[29]:


weather['WindSpeed9am'].isnull().sum()


# ### ALL MISSING VALUES HAVE BEEN RELACED,THUS DATA IS CLEANED OFF MISSING VALUE

# In[30]:


weather.isnull().sum()


# In[31]:


col=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustSpeed',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm']

for i in col:
    sb.distplot(weather[i])
    plt.show()


# In[32]:


sb.pairplot(weather,hue="RainToday", palette="hsv_r")


# In[33]:


Counter(weather['RainToday'])


# In[34]:


Counter(weather['RainTomorrow'])


# In[35]:


sb.countplot(weather['RainToday'])


# In[36]:


sb.countplot(weather['RainTomorrow'])


# In[37]:


def catgplot(weather,Categ,xcol='RainToday'):
    for i in Categ:
        print(i)
        sb.boxplot(xcol,i,data=weather)
        plt.xlabel(xcol)
        plt.ylabel([i])
        plt.show()
        
Categ = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
'Sunshine', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am','Temp3pm', 'RISK_MM']
catgplot(weather,Categ,xcol='RainToday')        


# In[38]:


def cvioplot(weather,Categ,xcol='RainTomorrow'):
    for i in Categ:
        print(i)
        sb.violinplot(xcol,i,data=weather)
        plt.xlabel(xcol)
        plt.ylabel([i])
        plt.show()
        
Categ = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
'Sunshine', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am','Temp3pm', 'RISK_MM']
cvioplot(weather,Categ,xcol='RainToday')   


# In[39]:


def swa(weather,col,hue,xcol='RISK_MM'):
    for i in col:
        print(i)
        sb.scatterplot(i,xcol,hue,data=weather)
        plt.xlabel(xcol)
        plt.ylabel([i])
        plt.legend()
        plt.show()
col=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustSpeed',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm']
palette = ['red','green','orange','blue']
swa(weather,col,hue='RainToday',xcol='RISK_MM')        


# In[40]:


def swa(weather,col,hue,xcol='RISK_MM'):
    for i in col:
        print(i)
        sb.scatterplot(i,xcol,hue,data=weather)
        plt.xlabel(xcol)
        plt.ylabel([i])
        plt.legend()
        plt.show()
col=['MinTemp', 'MaxTemp', 'Evaporation',
       'Sunshine', 'WindGustSpeed',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm']
palette = ['red','green','orange','blue']
swa(weather,col,hue='RainTomorrow',xcol='Rainfall') 


# In[41]:


def swa(weather,col,hue,xcol='RISK_MM'):
    for i in col:
        print(i)
        sb.lineplot(i,xcol,hue,data=weather)
        plt.xlabel(xcol)
        plt.ylabel([i])
        plt.legend()
        plt.show()
col=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustSpeed',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm']
palette = ['red','green','orange','blue']
swa(weather,col,hue='RainToday',xcol='RISK_MM')        


# In[42]:


def swa(weather,col,hue,xcol='Date'):
    for i in col:
        print(i)
        sb.lineplot(i,xcol,hue,data=weather)
        plt.xlabel(xcol)
        plt.ylabel([i])
        plt.legend()
        plt.show()
col=['MinTemp', 'MaxTemp', 'Evaporation',
       'Sunshine', 'WindGustSpeed',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm']
palette = ['red','green','orange','blue']
swa(weather,col,hue='RainTomorrow',xcol='Rainfall') 


# In[43]:


weather.corr()


# In[44]:


cor =weather.corr()
sb.heatmap(cor,xticklabels=cor.columns.values,yticklabels=cor.columns.values,annot=True,annot_kws={'size':7})


# ## MODELS FOR WEATHER PREDICTION

# In[85]:


from sklearn.preprocessing import scale,robust_scale,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,precision_recall_fscore_support
from sklearn.metrics import roc_auc_score,roc_curve


# In[46]:


weather.columns


# In[47]:


X = weather.drop(['Date', 'Location','RainToday', 'RainTomorrow'],axis=1)
print(X.shape)


# In[48]:


X = pd.get_dummies(X)
X = scale(X)


# In[49]:


y1 = weather['RainToday']
y2 = weather['RainTomorrow']
y1 = LabelEncoder().fit_transform(y1)
y2 = LabelEncoder().fit_transform(y2)
print(y1)
print('*'*100)
print(y2)


# 
# ### Modeling for RainToday with y1

# In[50]:


X_train,X_test,y_train,y_test = train_test_split(X,y1,test_size =.3,random_state=123,shuffle=True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# #### Logistic Regression

# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


log = LogisticRegression()
log = log.fit(X_train,y_train)
logpred = log.predict(X_test)
print('The accuracy for the logistic regression is {0}'.format(accuracy_score(logpred,y_test)* 100))


# In[53]:


print(classification_report(logpred,y_test))


# In[86]:


roc_auc_score(logpred,y_test)


# In[54]:


lconf = confusion_matrix(logpred,y_test)
conf = pd.DataFrame(lconf,('Not RainToday','RainToday'),('Not RainToday','RainToday'))
sb.heatmap(conf,xticklabels=conf.columns.values,yticklabels=conf.columns.values,annot=True,annot_kws={'size':12},cmap='icefire')


# In[90]:


fpr,tpr ,thresholds = roc_curve(logpred,y_test)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.plot(fpr,tpr)
plt.show()


# #### Random Forest

# In[55]:


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


# In[56]:


ran = RandomForestClassifier()
ran = ran.fit(X_train,y_train)
ranpred = ran.predict(X_test)


# In[57]:


print("The accuraacy for the random forest model is {0}".format(accuracy_score(ranpred,y_test)*100) )


# In[58]:


print(classification_report(ranpred,y_test))


# In[59]:


rconf = confusion_matrix(ranpred,y_test)
rrconf = pd.DataFrame(rconf,('Not RainToday','RainToday'),('Not RainToday','RainToday'))
sb.heatmap(rrconf,xticklabels=rrconf.columns.values,yticklabels=rrconf.columns.values,annot=True,annot_kws={'size':12},cmap='gist_heat_r')


# In[92]:


fpr,tpr ,thresholds = roc_curve(ranpred,y_test)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.plot(fpr,tpr)
plt.show()


# #### GradientBoosting Classifier

# In[77]:


Grad = GradientBoostingClassifier()
Grad = Grad.fit(X_train,y_train)
grapred = Grad.predict(X_test)
print('The accuracy for the GradientBoosting (for RainToday) is {0}'.format(accuracy_score(grapred,y_test)* 100))


# In[78]:


print(classification_report(gradpred,y_test))


# In[ ]:





# In[ ]:





# 
# ### Modeling for RainToday with y2

# In[61]:


x_train,x_test,y2_train,y2_test = train_test_split(X,y2,test_size =.3,random_state=123,shuffle=True)
print(x_train.shape)
print(y2_train.shape)
print(x_test.shape)
print(y2_test.shape)


# #### Logistic Regression

# In[62]:


log = LogisticRegression()
log = log.fit(x_train,y2_train)
ylogpred = log.predict(x_test)
print('The accuracy for the logistic regression is {0}'.format(accuracy_score(ylogpred,y2_test)* 100))


# In[63]:


print(classification_report(ylogpred,y2_test))


# In[64]:


print(confusion_matrix(ylogpred,y2_test))


# In[65]:


yconf = confusion_matrix(ylogpred,y2_test)
yyconf = pd.DataFrame(yconf,('Not RainToday','RainToday'),('Not RainToday','RainToday'))
sb.heatmap(yyconf,xticklabels=yyconf.columns.values,yticklabels=yyconf.columns.values,annot=True,annot_kws={'size':10},cmap='prism')


# #### Random forest

# In[66]:


yran = RandomForestClassifier()
yran = yran.fit(x_train,y2_train)
y2pred = yran.predict(x_test)
print('The accuracy for the random forest (for RainTomorrow) is {0}'.format(accuracy_score(y2pred,y2_test)* 100))


# In[67]:


print(classification_report(y2pred,y2_test))


# In[68]:


confusion_matrix(y2pred,y2_test)


# In[69]:


y2conf = confusion_matrix(y2pred,y2_test)
y22conf = pd.DataFrame(y2conf,('Not RainToday','RainToday'),('Not RainToday','RainToday'))
sb.heatmap(yyconf,xticklabels=y22conf.columns.values,yticklabels=y22conf.columns.values,annot=True,annot_kws={'size':10},cmap='magma')


# #### GradientBoosting Classifier

# In[79]:


yGrad = GradientBoostingClassifier()
yGrad = yGrad.fit(x_train,y2_train)
gGradpred = yGrad.predict(x_test)
print('The accuracy for the GradientBoosting (for RainTomorrow) is {0}'.format(accuracy_score(gradpred,y2_test)* 100))


# In[80]:


print(classification_report(gGradpred,y2_test))


# In[81]:


confusion_matrix(gGradpred,y2_test)


# In[82]:


grconf = confusion_matrix(gradpred,y2_test)
grdconf = pd.DataFrame(grconf,('Not RainToday','RainToday'),('Not RainToday','RainToday'))
sb.heatmap(grdconf,xticklabels=grdconf.columns.values,yticklabels=grdconf.columns.values,annot=True,annot_kws={'size':10},cmap='RdYlGn')


# ### MODEL PERFORMANCE FOR y1...ie RainToday 

# In[93]:


print('The accuracy for the Logistic Regression (for RainToday) is {0}'.format(accuracy_score(logpred,y_test)* 100))
print('The accuracy for the GradientBoosting (for RainToday) is {0}'.format(accuracy_score(ranpred,y_test)* 100))
print('The accuracy for the GradientBoosting (for RainToday) is {0}'.format(accuracy_score(grapred,y_test)* 100))


# ### MODEL PERFORMANCE FOR y2...ie RainTomorrow

# In[84]:


print('The accuracy for the Logistic Regression (for RainTomorrow) is {0}'.format(accuracy_score(ylogpred,y2_test)* 100))
print('The accuracy for the Random Forest(for RainTomorrow) is {0}'.format(accuracy_score(y2pred,y2_test)* 100))
print('The accuracy for the GradientBoosting (for RainTomorrow) is {0}'.format(accuracy_score(gGradpred,y2_test)* 100))


# In[ ]:




