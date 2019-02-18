#!/usr/bin/env python
# coding: utf-8

# ### Basic data analysis or exploratory data analysis (EDA)

# In[1]:


from __future__ import print_function
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read Training dataset as well as drop the index column
training_data = pd.read_csv('./data/cs-training.csv').drop('Unnamed: 0', axis = 1)


# For each column heading we replace "-" and convert the heading in lowercase 
cleancolumn = []
for i in range(len(training_data.columns)):
    cleancolumn.append(training_data.columns[i].replace('-', '').lower())
training_data.columns = cleancolumn


# In[3]:


# print the 5 records of the traiing dataset
training_data.head()


# In[4]:


# Describe the all statistical properties of the training dataset
training_data[training_data.columns[1:]].describe()


# In[5]:


training_data[training_data.columns[1:]].median()


# In[6]:


training_data[training_data.columns[1:]].mean()


# In[7]:


# This give you the calulation of the target lebels. Which category of the target lebel is how many percentage.
total_len = len(training_data['seriousdlqin2yrs'])
percentage_labels = (training_data['seriousdlqin2yrs'].value_counts()/total_len)*100
percentage_labels


# In[8]:


# Graphical representation of the target label percentage.
sns.set()
sns.countplot(training_data.seriousdlqin2yrs).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for seriousdlqin2yrs attribute")
ax.set_ylabel("Numbers of records")
plt.show()


# ### Missing values

# In[9]:


# You will get to know which column has missing value and it's give the count that how many records are missing 
training_data.isnull().sum()


# In[10]:


# Graphical representation of the missing values.
x = training_data.columns
y = training_data.isnull().sum()
sns.set()
sns.barplot(x,y)
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            int(height),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Data Attributes")
ax.set_ylabel("count of missing records for each attribute")
plt.xticks(rotation=90)
plt.show()


# In[11]:


# Actual replacement of the missing value using mean value.
training_data_mean_replace = training_data.fillna((training_data.mean()))
training_data_mean_replace.head()


# In[12]:


training_data_mean_replace.isnull().sum()


# In[13]:


# Actual replacement of the missing value using median value.
training_data_median_replace = training_data.fillna((training_data.median()))
training_data_median_replace.head()


# In[14]:


training_data_median_replace.isnull().sum()


# ### Correlation

# In[15]:


training_data.fillna((training_data.median()), inplace=True)
# Get the correlation of the training dataset
training_data[training_data.columns[1:]].corr()


# In[16]:


sns.set()
sns.set(font_scale=1.25)
sns.heatmap(training_data[training_data.columns[1:]].corr(),annot=True,fmt=".1f")
plt.show()


# ### Outliers Detection

# In[17]:


# Percentile based outlier detection
def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    (minval, maxval) = np.percentile(data, [diff, 100 - diff])
    #return minval, maxval
    return ((data < minval) | (data > maxval))
#percentile_based_outlier(data=training_data.revolvingutilizationofunsecuredlines)

# Another percentile based outlier detection method which is based on inter quertile(IQR) range
# import numpy as np
# def outliers_iqr(ys):
#     quartile_1, quartile_3 = np.percentile(ys, [25, 75])
#     iqr = quartile_3 - quartile_1
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((ys > upper_bound) | (ys < lower_bound))


# In[18]:


def mad_based_outlier(points, threshold=3.5):
    median_y = np.median(points)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in points])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in points]

    return np.abs(modified_z_scores) > threshold
#mad_based_outlier(points=training_data.age)


# In[19]:


def std_div(data, threshold=3):
    std = data.std()
    mean = data.mean()
    isOutlier = []
    for val in data:
        if val/std > threshold:
            isOutlier.append(True)
        else:
            isOutlier.append(False)
    return isOutlier
#std_div(data=training_data.age)


# In[20]:


def outlierVote(data):
    x = percentile_based_outlier(data)
    y = mad_based_outlier(data)
    z = std_div(data)
    temp = zip(data.index, x, y, z)
    final = []
    for i in range(len(temp)):
        if temp[i].count(False) >= 2:
            final.append(False)
        else:
            final.append(True)
    return final
#outlierVote(data=training_data.age)


# In[21]:


def plotOutlier(x):
    fig, axes = plt.subplots(nrows=4)
    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier, std_div, outlierVote]):
        sns.distplot(x, ax=ax, rug=True, hist=False)
        outliers = x[func(x)]
        ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top', size=20)
    axes[0].set_title('Percentile-based Outliers', **kwargs)
    axes[1].set_title('MAD-based Outliers', **kwargs)
    axes[2].set_title('STD-based Outliers', **kwargs)
    axes[3].set_title('Majority vote based Outliers', **kwargs)
    fig.suptitle('Comparing Outlier Tests with n={}'.format(len(x)), size=20)
    fig = plt.gcf()
    fig.set_size_inches(15,10)


# In[22]:


plotOutlier(training_data.revolvingutilizationofunsecuredlines.sample(5000))


# In[23]:


plotOutlier(training_data.age.sample(1000))


# In[24]:


plotOutlier(training_data.numberoftime3059dayspastduenotworse.sample(1000))


# In[25]:


plotOutlier(training_data.debtratio.sample(1000))


# In[26]:


plotOutlier(training_data.monthlyincome.sample(1000))


# In[27]:


plotOutlier(training_data.numberofopencreditlinesandloans.sample(1000))


# In[28]:


plotOutlier(training_data.numberoftimes90dayslate.sample(1000))


# In[29]:


plotOutlier(training_data.numberrealestateloansorlines.sample(1000))


# In[30]:


plotOutlier(training_data.numberoftime6089dayspastduenotworse.sample(1000))


# In[31]:


plotOutlier(training_data.numberofdependents.sample(1000))


# ### Handle the outliers

# In[32]:


revNew = []
training_data.revolvingutilizationofunsecuredlines
for val in training_data.revolvingutilizationofunsecuredlines:
    if val <= 0.99999:
        revNew.append(val)
    else:
        revNew.append(0.99999)
training_data.revolvingutilizationofunsecuredlines = revNew


# In[33]:


training_data.age.plot.box()


# In[34]:


import collections
collections.Counter(training_data.age)


# In[35]:


ageNew = []
for val in training_data.age:
    if val > 21:
        ageNew.append(val)
    else:
        ageNew.append(21)
        
training_data.age = ageNew


# In[36]:


collections.Counter(training_data.numberoftime3059dayspastduenotworse)


# In[37]:


New = []
med = training_data.numberoftime3059dayspastduenotworse.median()
for val in training_data.numberoftime3059dayspastduenotworse:
    if ((val == 98) | (val == 96)):
        New.append(med)
    else:
        New.append(val)

training_data.numberoftime3059dayspastduenotworse = New


# In[38]:


def outlierRatio(data):
    functions = [percentile_based_outlier, mad_based_outlier, std_div, outlierVote]
    outlierDict = {}
    for func in functions:
        funcResult = func(data)
        count = 0
        for val in funcResult:
            if val == True:
                count += 1 
        outlierDict[str(func)[10:].split()[0]] = [count, '{:.2f}%'.format((float(count)/len(data))*100)]
    
    return outlierDict
outlierRatio(training_data.debtratio)


# In[39]:


plotOutlier(training_data.debtratio.sample(1000))


# In[40]:


def add_freq():
    ncount = len(training_data)

    ax2=ax.twinx()

    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom')

    ax2.set_ylim(0,100)
    ax2.grid(None)
ax = sns.countplot(mad_based_outlier(training_data.debtratio))

add_freq()


# In[41]:


minUpperBound = min([val for (val, out) in zip(training_data.debtratio, mad_based_outlier(training_data.debtratio)) if out == True])


# In[42]:


newDebtRatio = []
for val in training_data.debtratio:
    if val > minUpperBound:
        newDebtRatio.append(minUpperBound)
    else:
        newDebtRatio.append(val)

training_data.debtratio = newDebtRatio 


# In[43]:


def plotOutlierFree(x):
    fig, axes = plt.subplots(nrows=4)
    nOutliers = []
    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier, std_div, outlierVote]):
        tfOutlier = zip(x, func(x))
        nOutliers.append(len([index for (index, bol) in tfOutlier if bol == True]))
        outlierFree = [index for (index, bol) in tfOutlier if bol == True]
        sns.distplot(outlierFree, ax=ax, rug=True, hist=False)
        
    kwargs = dict(y=0.95, x=0.05, ha='left', va='top', size=15)
    axes[0].set_title('Percentile-based Outliers, removed: {r}'.format(r=nOutliers[0]), **kwargs)
    axes[1].set_title('MAD-based Outliers, removed: {r}'.format(r=nOutliers[1]), **kwargs)
    axes[2].set_title('STD-based Outliers, removed: {r}'.format(r=nOutliers[2]), **kwargs)
    axes[3].set_title('Majority vote based Outliers, removed: {r}'.format(r=nOutliers[3]), **kwargs)
    fig.suptitle('Outlier Removed By Method with n={}'.format(len(x)), size=20)
    fig = plt.gcf()
    fig.set_size_inches(15,10)


# In[44]:


plotOutlierFree(training_data.monthlyincome.sample(1000))


# In[45]:


def replaceOutlier(data, method = outlierVote, replace='median'):
    '''replace: median (auto)
                'minUpper' which is the upper bound of the outlier detection'''
    vote = outlierVote(data)
    x = pd.DataFrame(zip(data, vote), columns=['debt', 'outlier'])
    if replace == 'median':
        replace = x.debt.median()
    elif replace == 'minUpper':
        replace = min([val for (val, vote) in zip(data, vote) if vote == True])
        if replace < data.mean():
            return 'There are outliers lower than the sample mean'
    debtNew = []
    for i in range(x.shape[0]):
        if x.iloc[i][1] == True:
            debtNew.append(replace)
        else:
            debtNew.append(x.iloc[i][0])
    
    return debtNew


# In[46]:


incomeNew = replaceOutlier(training_data.monthlyincome, replace='minUpper')


# In[47]:


training_data.monthlyincome = incomeNew


# In[48]:


collections.Counter(training_data.numberoftimes90dayslate)


# In[49]:


def removeSpecificAndPutMedian(data, first = 98, second = 96):
    New = []
    med = data.median()
    for val in data:
        if ((val == first) | (val == second)):
            New.append(med)
        else:
            New.append(val)
            
    return New


# In[50]:


new = removeSpecificAndPutMedian(training_data.numberoftimes90dayslate)


# In[51]:


training_data.numberoftimes90dayslate = new


# In[52]:


collections.Counter(training_data.numberrealestateloansorlines)


# In[53]:


realNew = []
for val in training_data.numberrealestateloansorlines:
    if val > 17:
        realNew.append(17)
    else:
        realNew.append(val)
training_data.numberrealestateloansorlines = realNew


# In[54]:


collections.Counter(training_data.numberoftime6089dayspastduenotworse)


# In[55]:


new = removeSpecificAndPutMedian(training_data.numberoftime6089dayspastduenotworse)
training_data.numberoftime6089dayspastduenotworse = new


# In[56]:


collections.Counter(training_data.numberofdependents)


# In[57]:


depNew = []
for var in training_data.numberofdependents:
    if var > 10:
        depNew.append(10)
    else:
        depNew.append(var)


# In[58]:


training_data.numberofdependents = depNew


# ### Feature Importance

# In[59]:


training_data.head()


# In[60]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


training_data.columns[1:]


# In[62]:


X = training_data.drop('seriousdlqin2yrs', axis=1)
y = training_data.seriousdlqin2yrs
features_label = training_data.columns[1:]
forest = RandomForestClassifier (n_estimators = 10000, random_state=0, n_jobs = -1)
forest.fit(X,y)
importances = forest.feature_importances_
indices = np. argsort(importances)[::-1]
for i in range(X.shape[1]):
    print ("%2d) %-*s %f" % (i + 1, 30, features_label[i],importances[indices[i]]))


# In[63]:


plt.title('Feature Importances')
plt.bar(range(X.shape[1]),importances[indices], color="green", align="center")
plt.xticks(range(X.shape[1]),features_label, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# ## Train and build baseline model

# In[64]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[65]:


X = training_data.drop('seriousdlqin2yrs', axis=1)
y = training_data.seriousdlqin2yrs


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[67]:


knMod = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                             metric='minkowski', metric_params=None)


# In[68]:


knMod.fit(X_train, y_train)


# In[69]:


knMod.score(X_test, y_test)


# In[70]:


test_labels=knMod.predict_proba(np.array(X_test.values))[:,1]


# In[71]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[72]:


glmMod = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                            intercept_scaling=1, class_weight=None, 
                            random_state=None, solver='liblinear', max_iter=100,
                            multi_class='ovr', verbose=2)


# In[73]:


glmMod.fit(X_train, y_train)


# In[74]:


glmMod.score(X_test, y_test)


# In[75]:


test_labels=glmMod.predict_proba(np.array(X_test.values))[:,1]


# In[76]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[77]:


adaMod = AdaBoostClassifier(base_estimator=None, n_estimators=200, learning_rate=1.0)


# In[78]:


adaMod.fit(X_train, y_train)


# In[79]:


adaMod.score(X_test, y_test)


# In[80]:


test_labels=adaMod.predict_proba(np.array(X_test.values))[:,1]


# In[81]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[82]:


gbMod = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0,
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                   max_depth=3,
                                   init=None, random_state=None, max_features=None, verbose=0)


# In[83]:


gbMod.fit(X_train, y_train)


# In[84]:


gbMod.score(X_test, y_test)


# In[85]:


test_labels=gbMod.predict_proba(np.array(X_test.values))[:,1]


# In[86]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[87]:


rfMod = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                               max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
                               random_state=None, verbose=0)


# In[88]:


rfMod.fit(X_train, y_train)


# In[89]:


rfMod.score(X_test, y_test)


# In[90]:


test_labels=rfMod.predict_proba(np.array(X_test.values))[:,1]


# In[91]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# ### Cross Validation

# In[92]:


from sklearn.model_selection import cross_val_score
def cvDictGen(functions, scr, X_train=X, y_train=y, cv=5, verbose=1):
    cvDict = {}
    for func in functions:
        cvScore = cross_val_score(func, X_train, y_train, cv=cv, verbose=verbose, scoring=scr)
        cvDict[str(func).split('(')[0]] = [cvScore.mean(), cvScore.std()]
    
    return cvDict

def cvDictNormalize(cvDict):
    cvDictNormalized = {}
    for key in cvDict.keys():
        for i in cvDict[key]:
            cvDictNormalized[key] = ['{:0.2f}'.format((cvDict[key][0]/cvDict[cvDict.keys()[0]][0])),
                                     '{:0.2f}'.format((cvDict[key][1]/cvDict[cvDict.keys()[0]][1]))]
    return cvDictNormalized


# In[93]:


cvD = cvDictGen(functions=[knMod, glmMod, adaMod, gbMod, rfMod], scr='roc_auc')
cvD


# ### Hyper parameter optimization using Randomized search

# In[94]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# #### AdaBoost

# In[95]:


adaHyperParams = {'n_estimators': [10,50,100,200,420]}


# In[96]:


gridSearchAda = RandomizedSearchCV(estimator=adaMod, param_distributions=adaHyperParams, n_iter=5,
                                   scoring='roc_auc', fit_params=None, cv=None, verbose=2).fit(X_train, y_train)


# In[97]:


gridSearchAda.best_params_, gridSearchAda.best_score_


# #### GradientBoosting

# In[98]:


gbHyperParams = {'loss' : ['deviance', 'exponential'],
                 'n_estimators': randint(10, 500),
                 'max_depth': randint(1,10)}


# In[99]:


gridSearchGB = RandomizedSearchCV(estimator=gbMod, param_distributions=gbHyperParams, n_iter=10,
                                   scoring='roc_auc', fit_params=None, cv=None, verbose=2).fit(X_train, y_train)


# In[100]:


gridSearchGB.best_params_, gridSearchGB.best_score_


# ### Train models with help of new hyper parameter

# In[101]:


bestGbModFitted = gridSearchGB.best_estimator_.fit(X_train, y_train)


# In[102]:


bestAdaModFitted = gridSearchAda.best_estimator_.fit(X_train, y_train)


# In[103]:


cvDictbestpara = cvDictGen(functions=[bestGbModFitted, bestAdaModFitted], scr='roc_auc')


# In[104]:


cvDictbestpara


# In[105]:


test_labels=bestGbModFitted.predict_proba(np.array(X_test.values))[:,1]


# In[106]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[107]:


test_labels=bestAdaModFitted.predict_proba(np.array(X_test.values))[:,1]


# In[108]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# ### Feature Transformation

# In[110]:


import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
X_train_1 = np.array(X_train)
X_train_transform = transformer.transform(X_train_1)


# In[111]:


bestGbModFitted_transformed = gridSearchGB.best_estimator_.fit(X_train_transform, y_train)


# In[112]:


bestAdaModFitted_transformed = gridSearchAda.best_estimator_.fit(X_train_transform, y_train)


# In[113]:


cvDictbestpara_transform = cvDictGen(functions=[bestGbModFitted_transformed, bestAdaModFitted_transformed],
                                     scr='roc_auc')


# In[114]:


cvDictbestpara_transform


# In[115]:


import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
X_test_1 = np.array(X_test)
X_test_transform = transformer.transform(X_test_1)


# In[118]:


X_test_transform


# In[119]:


test_labels=bestGbModFitted_transformed.predict_proba(np.array(X_test_transform))[:,1]


# In[120]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[121]:


test_labels=bestAdaModFitted_transformed.predict_proba(np.array(X_test_transform))[:,1]


# In[122]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# ### Voting based ensamble model

# In[139]:


from sklearn.ensemble import VotingClassifier
votingMod = VotingClassifier(estimators=[('gb', bestGbModFitted_transformed), 
                                         ('ada', bestAdaModFitted_transformed)], voting='soft',weights=[2,1])
votingMod = votingMod.fit(X_train_transform, y_train)


# In[140]:


test_labels=votingMod.predict_proba(np.array(X_test_transform))[:,1]


# In[141]:


votingMod.score(X_test_transform, y_test)


# In[142]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# In[143]:


from sklearn.ensemble import VotingClassifier
votingMod_old = VotingClassifier(estimators=[('gb', bestGbModFitted), ('ada', bestAdaModFitted)], 
                                 voting='soft',weights=[2,1])
votingMod_old = votingMod.fit(X_train, y_train)


# In[144]:


test_labels=votingMod_old.predict_proba(np.array(X_test.values))[:,1]


# In[145]:


roc_auc_score(y_test,test_labels , average='macro', sample_weight=None)


# ### Testing on Real Test Dataset

# In[163]:


# Read Training dataset as well as drop the index column
test_data = pd.read_csv('./data/cs-test.csv').drop('Unnamed: 0', axis = 1)
# For each column heading we replace "-" and convert the heading in lowercase 
cleancolumn = []
for i in range(len(test_data.columns)):
    cleancolumn.append(test_data.columns[i].replace('-', '').lower())
test_data.columns = cleancolumn


# In[164]:


test_data.drop(['seriousdlqin2yrs'], axis=1, inplace=True)
test_data.fillna((training_data.median()), inplace=True)


# In[165]:


test_labels_votingMod_old = votingMod_old.predict_proba(np.array(test_data.values))[:,1]
print (len(test_labels_votingMod_old))


# In[166]:


output = pd.DataFrame({'ID':test_data.index, 'probability':test_labels_votingMod_old})


# In[167]:


output.to_csv("./predictions.csv", index=False)


# In[168]:


import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
test_data_temp = np.array(test_data)
test_data_transform = transformer.transform(test_data_temp)


# In[169]:


test_labels_votingMod = votingMod.predict_proba(np.array(test_data.values))[:,1]
print (len(test_labels_votingMod_old))


# In[170]:


output = pd.DataFrame({'ID':test_data.index, 'probability':test_labels_votingMod})


# In[171]:


output.to_csv("./predictions_voting_Feature_transformation.csv", index=False)


# In[ ]:




