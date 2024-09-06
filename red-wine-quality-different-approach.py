#!/usr/bin/env python
# coding: utf-8

# This work is taken from Nima Pourmoradi

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')

print(colored('\nAll libraries imported succesfully.', 'green'))


# ### Library Configuration

# In[2]:


pd.options.mode.copy_on_write = True # Allow re-write on variable
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# ### Reading Data

# In[3]:


df = pd.read_csv('/kaggle/input/red-wine-quality/winequality-red.csv',sep=';')
df.head()


# In[4]:


df.info()


# In[5]:


df.describe().T.style.background_gradient(axis=0)


# ### Missing Values

# In[6]:


df.isna().sum()


# ### Data Visualization

# In[7]:


df.rename(columns = {"fixed acidity": "fixed_acidity",
                       "volatile acidity": "volatile_acidity",
                       "citric acid": "citric_acid",
                       "residual sugar": "residual_sugar",
                       "chlorides": "chlorides",
                       "free sulfur dioxide": "free_sulfur_dioxide",
                       "total sulfur dioxide": "total_sulfur_dioxide"},
            inplace = True)


# In[8]:


columns = list(df.columns)
columns


# In[9]:


fig, ax = plt.subplots(11,2,figsize=(15,45))
plt.subplots_adjust(hspace=0.5)
for i in range(11):
    # AX 1
    sns.boxplot(x=columns[i],data=df,ax=ax[i,0])
    # AX 2
    sns.scatterplot(x=columns[i], y='quality', data=df, hue='quality', ax=ax[i,1])


# In[10]:


corr = df.corr()
plt.figure(figsize=(9,6))
sns.heatmap(corr, annot=True, fmt='.2f', linewidth=0.5, cmap='Purples', mask = np.triu(corr))
plt.show()


# In[11]:


sns.pairplot(df, hue='quality',corner=True,palette='Purples')


# In[12]:


# Best Correlations are between 
# citeic_acid & flex_acidity ---> 0.67
# density & flex_acidity ---> 0.67
# total_sulfor_dioxide & free_sulfor_dioxide ---> 0.67


# In[13]:


df.quality.unique()


# In[14]:


df = df.replace({'quality' : {
                                    8 : 'Good',
                                    7 : 'Good',
                                    6 : 'Middle',
                                    5 : 'Middle',
                                    4 : 'Bad',
                                    3 : 'Bad',
        }}
)


# In[15]:


df.head()


# ## Normalization

# In[16]:


X = df.drop(columns='quality')
y = df.quality


# In[17]:


scaler = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
X_scaled = pd.DataFrame(scaler, columns=X.columns)
X_scaled.describe().T.style.background_gradient(axis=0, cmap='Purples')


# ### Initialization

# In[18]:


# define a function to ploting Confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    '''Ploting Confusion Matrix'''
    cm = metrics.confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    ax = sns.heatmap(cm, annot=True, fmt='',cmap='Purples')
    ax.set_xlabel('Predicted Labels', fontsize=18)
    ax.set_ylabel('True Labels', fontsize=18)
    ax.set_title('Confusion Matrix', fontsize=25)
    ax.xaxis.set_ticklabels(['Bad', 'Middle','Good'])
    ax.yaxis.set_ticklabels(['Bad', 'Middle','Good'])
    plt.show()


# In[19]:


# define a function to ploting Classification report
def clfr_plot(y_test,y_pred):
    ''' Plotting Classification Report'''
    cr = pd.DataFrame(metrics.classification_report(y_test, y_pred_rf, digits=3, output_dict=True)).T
    cr.drop(columns='support', inplace=True)
    sns.heatmap(cr, annot=True, cmap='Purples',linecolor='white',linewidth=0.5).xaxis.tick_top()


# In[20]:


def clf_plot(y_pred):
    '''1) Ploting Confusion Matrix
       2) Plotting Classification Report'''
    cm = metrics.confusion_matrix(y_test,y_pred)
    cr = pd.DataFrame(metrics.classification_report(y_test, y_pred_rf, digits=3, output_dict=True)).T
    cr.drop(columns='support', inplace=True)
    
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    
    # Left AX : Confusion Matrix
    
    ax[0]=sns.heatmap(cm, annot=True, fmt='', cmap='Purples',ax=ax[0])
    ax[0].set_xlabel('Predicted Labels', fontsize=18)
    ax[0].set_ylabel('True Labels', fontsize=18)
    ax[0].set_title('Confusion Matrix', fontsize=25)
    ax[0].xaxis.set_ticklabels(['Bad', 'Middle','Good'])
    ax[0].yaxis.set_ticklabels(['Bad', 'Middle','Good'])
    
    # Right AX : Classification Report
    ax[1]=sns.heatmap(cr, annot=True, cmap='Purples',linecolor='white',linewidth=0.5)
    ax[1].xaxis.tick_top()
    ax[1].set_title('Classification Report', fontsize=25)
    
    plt.show()


# In[21]:


df.quality.value_counts()


# In[22]:


## Split DataFrame To train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)


# ### First Model : Random Forest Classifier

# In[23]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'n_estimators':[50,150,500],
    'criterion':['gini','entropy','log_loss'],
    'max_features' : ['sqrt','log2']
}

rf = RandomForestClassifier(n_jobs=-1) # parameter that makes all processors available to perform the operation
rf_cv = GridSearchCV(estimator=rf, cv=20, param_grid=parameters).fit(X_train, y_train) # finds the best combination of hyperparameters

print('Tuned hyper parameters :',rf_cv.best_params_)
print('accureacy :', rf_cv.best_score_)


# In[24]:


# Model
rf = RandomForestClassifier(**rf_cv.best_params_).fit(X_train,y_train)


# In[25]:


y_pred_rf = rf.predict(X_test)

rf_score = round(rf.score(X_test,y_test),3)
print('RandomForestClassifier Score :', rf_score)


# In[26]:


y_test.value_counts()


# In[27]:


clf_plot(y_pred_rf)


# ### Second Model : Logistic Regression

# In[28]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'class_weight' : ['balanced'],
    'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

lr = LogisticRegression()
lr_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10).fit(X_train, y_train)

print('Tuned hyper parameters : ', lr_cv.best_params_)
print('accuracy : ', lr_cv.best_score_)


# In[29]:


lr = LogisticRegression(**lr_cv.best_params_).fit(X_train, y_train)


# In[30]:


y_pred_lr = lr.predict(X_test)

lr_score = round(lr.score(X_test,y_test),3)
print('Logistic Regression Score :' , lr_score)


# In[31]:


clf_plot(y_pred_lr)


# ## Third Model : SVC

# In[32]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'gamma' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
}



svc = SVC()
svc_cv = GridSearchCV(estimator=svc, param_grid=parameters, cv=10).fit(X_train, y_train)



print('Tuned hyper parameters : ', svc_cv.best_params_)
print('accuracy : ', svc_cv.best_score_)


# In[33]:


svc = SVC(**svc_cv.best_params_).fit(X_train,y_train)


# In[34]:


y_pred_svc = svc.predict(X_test)

svc_score = round(svc.score(X_test,y_test),3)
print('SVC Score :', svc_score)


# In[35]:


clf_plot(y_pred_svc)


# ## Fourth Model : Decision Tree Classifier

# In[36]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'splitter' : ['best', 'random'],
    'max_depth' : list(np.arange(4, 30, 1))
        }



tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(estimator=tree, cv=10, param_grid=parameters).fit(X_train, y_train)



print('Tuned hyper parameters : ', tree_cv.best_params_)
print('accuracy : ', tree_cv.best_score_)


# In[37]:


tree = DecisionTreeClassifier(**tree_cv.best_params_).fit(X_train,y_train)


# In[38]:


y_pred_tree = tree.predict(X_test)

tree_score = round(tree.score(X_test,y_test),3)
print('Decision Tree Classifier Score :', tree_score)


# In[39]:


clf_plot(y_pred_tree)


# ## Fifth Model : KNeighborsClassifier

# In[40]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'n_neighbors' : list(np.arange(3, 50, 2)),
    'weights': ['uniform', 'distance'],
    'p' : [1, 2, 3, 4]
}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(estimator=knn, cv=10, param_grid=parameters).fit(X_train, y_train)

print('Tuned hyper parameters : ', knn_cv.best_params_)
print('accuracy : ', knn_cv.best_score_)


# In[41]:


knn = KNeighborsClassifier(**knn_cv.best_params_).fit(X_train,y_train)


# In[42]:


y_pred_knn = knn.predict(X_test)

knn_score = round(knn.score(X_test,y_test),3)
print('KNeighborsClassifier Score :',knn_score)


# In[43]:


clf_plot(y_pred_knn)


# ## Sixth Model : GaussianNB

# In[44]:


gnb = GaussianNB().fit(X_train,y_train)
y_pred_gnb = gnb.predict(X_test)
gnb_score = round(gnb.score(X_test,y_test),3)
print('GaussianNB Score :',gnb_score)


# In[45]:


clf_plot(y_pred_gnb)


# ## Result

# In[46]:


result = pd.DataFrame({
    'Algorithm' : ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'GaussianNB'],
    'Score' : [rf_score, lr_score, svc_score, tree_score, knn_score, gnb_score]
})

result.sort_values(by='Score', inplace=True)


# In[47]:


sns.set_palette("Purples")


# In[48]:


fig, ax = plt.subplots(1,1,figsize=(15,5))

sns.barplot(x='Algorithm', y='Score', data=result)
ax.bar_label(ax.containers[0], fmt='%.3f')
ax.set_xticklabels(labels=result.Algorithm, rotation=300)
plt.show()


# In[ ]:




