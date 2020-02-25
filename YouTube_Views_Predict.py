# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:27:03 2020

@author: Shivam Kolhe
DATASET: YouTube Views Prediction
HOSTED BY: Skillenza
File: 1
Minimum RMSE Achieved : 7.62
Best Algorithm : LightGBM
"""

"""
## Tasks to do : 
1) handle '"24'
2) Outliers
   -- Transformation
3) Look for other flaws in categorical variables
4) Dates (fetch Years))
5) Convert to LOWERCASE
6) Null Values
7) Data type conversion
8) LabelEncoding
9) Dummies
10) Scaling
11) Correlation
"""

""" Models Applied
1) Lasso
2) Ridge
3) ElasticNet
4) LightGBM
5) Kernal Ridge
6) XGBoost
7) Random Forest Regressor
8) Stacking Regressor

"""

"""  *********************************** Some Important Functions ******************************* """
def ShowNullPercentage(data, sortType=False):
        Columns = list(data.columns)
        Percentage = []
        for val in data:
            Percentage.append((data[val].isnull().sum() / data.shape[0]) * 100)
        
        nulldf = {'Column':Columns, 'Percentage':Percentage}
        return(pd.DataFrame(nulldf).sort_values(['Percentage'], ascending=sortType).to_string())
        
def imputeTrainNull(data):    
    data.Trend_day_count = pd.to_numeric(data.Trend_day_count, errors='coerce')
    data.Tag_count = pd.to_numeric(data.Tag_count, errors='coerce')
    data.likes = pd.to_numeric(data.likes, errors='coerce')
    data.dislike = pd.to_numeric(data.dislike, errors='coerce')
    data.views = pd.to_numeric(data.views, errors='coerce')
    data.comment_count = pd.to_numeric(data.comment_count, errors='coerce')
    data.Trend_tag_count = pd.to_numeric(data.Trend_tag_count, errors='coerce')
    
    data.comment_disabled.fillna(method ='ffill', inplace=True)
    data['like dislike disabled'].fillna(method ='ffill', inplace=True)
    data.category_id.fillna(method ='ffill', inplace=True)  
    data.Trend_day_count.fillna(data.Trend_day_count.median(), inplace=True)
    data.Tag_count.fillna(data.Tag_count.median(), inplace=True)
    data.Trend_tag_count.fillna(data.Trend_tag_count.median(), inplace=True)
    data['tag appered in title'].fillna(method ='ffill', inplace=True)
    data.comment_count.fillna(data.comment_count.median(), inplace=True)
    data.likes.fillna(data.likes.median(), inplace=True)
    data.dislike.fillna(data.dislike.median(), inplace=True)
    data.views.fillna(data.views.median(), inplace=True)
    data.subscriber.fillna(data.subscriber.median(), inplace=True) 
    print(data.isnull().sum())
    

def imputeTestNull(data):
    data.Trend_day_count = pd.to_numeric(data.Trend_day_count, errors='coerce')
    data.Tag_count = pd.to_numeric(data.Tag_count, errors='coerce')
    data.likes = pd.to_numeric(data.likes, errors='coerce')
    data.dislike = pd.to_numeric(data.dislike, errors='coerce')
    data.comment_count = pd.to_numeric(data.comment_count, errors='coerce')
    data.Trend_tag_count = pd.to_numeric(data.Trend_tag_count, errors='coerce')
    data.comment_disabled.fillna(method ='ffill', inplace=True)
    data['like dislike disabled'].fillna(method ='ffill', inplace=True)
    
    data.category_id.fillna(method ='ffill', inplace=True)  
    data.comment_count.fillna(data.comment_count.median(), inplace=True)
    # Imputing Null Values by Median
    data.subscriber.fillna(data.subscriber.median(), inplace=True) 
    print(data.isnull().sum())
    
    
def handleOutliers(data):
    df_num = data.select_dtypes(include=['int64','float64' ])
    z = np.abs(stats.zscore(df_num))
    #threshold = 3
    x= np.where(z > 3)
    data.drop(data.index[x[:1]], inplace=True)
    print(df_num.skew())

""" ******************************************************************************************* """


###########################################################################################################
###########################################################################################################
################################### ---------- START ----------- ##########################################
###########################################################################################################
###########################################################################################################

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')         # Reading Training File

test = pd.read_csv('test.csv')           # Reading Test File
#testid = test.iloc[:,0]
###############################################################################

#print(st.ShowNullPercentage(train))
#train = train.iloc[:,0:19]
#train = imputeTrainNull(train)
#train.isnull().sum()
train = train.iloc[:,0:19]
train.drop(['Video_id', 'channel_title', 'title', 'tags', 'description'], axis=1, inplace=True)
test.drop(['Video_id', 'channel_title', 'title', 'tags', 'description'], axis=1, inplace=True)


# Null Value Imputation from train
train.isnull().sum()
imputeTrainNull(train)

# Handling Outliers from Train Data
handleOutliers(train)
train.skew()
############################## Merging Dataset (Train and Test) ################################
ntrain = train.shape[0]
ntest = test.shape[0]
train.views = ((train.views) ** (1/3))
y_train = train.views
del train['views']
data = pd.concat((train, test)).reset_index(drop=True)
print("Data size is : {}".format(data.shape))

data.publish_date.fillna(method ='ffill', inplace=True)  
del data['publish_date']
data.trending_date.fillna(method ='ffill', inplace=True)  


for i in range(data.trending_date.shape[0]):
    #data.publish_date[i] = data.publish_date[i].split('-')[0]
    data.trending_date[i] = data.trending_date[i].split('-')[0]

#################################################################################################
data.category_id.unique()
data[data.category_id=='“24']
data.category_id = data.category_id.replace({'“24':'24'})

data.comment_disabled.unique()
data[data.comment_disabled=='25']
data.comment_disabled = data.comment_disabled.replace({'25':None})

data['like dislike disabled'].unique()
data['like dislike disabled'] = data['like dislike disabled'].replace({'4':None})

data['tag appered in title'].unique()
###############################################################################

############################################################################################
data.isnull().sum()
# Null Value Imputation from data
imputeTestNull(data)  
data.skew()   
data.subscriber = ((data.subscriber) ** (1/3))    # Cube root transformation of Scuscriber
############################################################################################

################################# Checking for Dominating Classes ####################################
def ShowDominatingPercentage(data, sortType=False):
        columns = data.columns
        perc = []
        for feature in data.columns:
            dominant = (data[feature].value_counts() / np.float(len(data))).sort_values(ascending=False).values[0] 
            perc.append(dominant*100)
        uniquedf = {'Column':columns, 'Percentage':perc}
        return(pd.DataFrame(uniquedf).sort_values(['Percentage'], ascending=sortType).to_string())
print(ShowDominatingPercentage(data))
######################################################################################################


################################# CORRPLOT ###########################################################
corrmat = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
# Corrplot shows high correlation among (likes and dislike) and (subscriber and views)

data.drop(['dislike'], axis=1, inplace=True)            # Dropping dislike column
######################################################################################################

# Converting Data Types
data.info()
data['tag appered in title'] = data['tag appered in title'].astype('object')
data['comment_disabled'] = data['comment_disabled'].astype('str')
data['like dislike disabled'] = data['like dislike disabled'].astype('str')

############################# Converting to Classes to Lower Case ############################
data.comment_disabled = data.comment_disabled.str.lower()
data['like dislike disabled'] = data['like dislike disabled'].str.lower()    
##############################################################################################

############################### Creating Dummies ##############################
# Cleaning Category_id
data.category_id = pd.to_numeric(data.category_id, errors='coerce')
data.category_id = data.category_id.astype('str')
data.category_id = data.category_id.replace({'224.0':'24.0'})
data.category_id = data.category_id.replace({'2225.0':'25.0'})
data.category_id = data.category_id.replace({'226.0':'26.0'})
data.category_id = data.category_id.replace({'210.0':'10.0'})
data.category_id = data.category_id.replace({'117.0':'17.0'})
data.category_id = data.category_id.replace({'122.0':'22.0'})
data.category_id.unique()

dt1=pd.get_dummies(data,drop_first=True)     # Dummies
###############################################################################

############################## Feature Scaling ################################
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
dt1.iloc[:,0:6] = sc_X.fit_transform(dt1.iloc[:,0:6])
data_encoded_scale = pd.DataFrame(dt1)
###############################################################################

######################## Separating Train and Test ############################
train = data_encoded_scale[:ntrain]
test = data_encoded_scale[ntrain:]
y_train
###############################################################################

###############################################################################
###############################################################################
#################### Applying Machine Learning Models #########################
###############################################################################
###############################################################################

################################## Libraries ##################################
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
###############################################################################


################################## Lasso Reg ##################################
tuned_paramaters = [{'alpha':[5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.05,0.001],
                     'max_iter':[100,500,1000,1500,2000]}]
    
lasso = Lasso()
regrL = GridSearchCV(lasso, tuned_paramaters, cv = 5, scoring='neg_mean_squared_error')
regrL.fit(train, y_train)
print(np.sqrt(-regrL.best_score_))
regrL.best_estimator_
regrL.best_params_
lasso_new = Lasso(alpha=0.01, max_iter=500)
###############################################################################


################################## Ridge Reg ##################################
tuned_paramaters = [{'alpha':[14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5, 17,19,20],
                     'max_iter':[100,500,1000,1500,2000]}]
    
ridge = Ridge()
regr = GridSearchCV(ridge, tuned_paramaters, cv = 5, scoring='neg_mean_squared_error')
regr.fit(train, y_train)
print(np.sqrt(-regr.best_score_))
regr.best_estimator_
regr.best_params_
ridge_new = Ridge(alpha=14.5, max_iter=100)
###############################################################################


############################# Elastic Regression ##############################
tuned_paramaters = [{'alpha':[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007],
                     'l1_ratio':[0.8, 0.85, 0.9, 0.95, 0.99, 1],
                     'max_iter':[100,500,1000,1500,2000]}]

elastic = ElasticNet()    
regrE = GridSearchCV(elastic, tuned_paramaters, cv = 5, scoring='neg_mean_squared_error')
regrE.fit(train, y_train)
print(np.sqrt(-regrE.best_score_))
regrE.best_estimator_
regrE.best_params_

elastic_new = ElasticNet(alpha=0.0007, l1_ratio=0.8, max_iter=1500)
###############################################################################

############################### XGB Regression ################################
tuned_paramaters = [{'n_estimators':[150,350,650,850,1000],
                     'learning_rate':[0.01,0.1,0.2,0.3],
                     'max_depth':[5,10,15,20]}]

model_xgb = xgb.XGBRegressor() #the params 
regrX = GridSearchCV(model_xgb, tuned_paramaters, cv = 2, scoring='neg_mean_squared_error')
regrX.fit(train, y_train)
print(np.sqrt(-regrX.best_score_))
regrX.best_estimator_
regrX.best_params_  
ypredX = regrX.predict(test)

ypredXc = ypredX ** 3

df = pd.DataFrame({'views':ypredXc})
df.to_csv('submodXGB2.csv') 

###############################################################################
xgboost = xgb.XGBRegressor(learning_rate=0.01,n_estimators=1000,
                     max_depth=15, min_child_weight=0,
                     gamma=0, subsample=0.7,
                     colsample_bytree=0.7,
                     objective='reg:linear', nthread=-1,
                     scale_pos_weight=1, seed=27,
                     reg_alpha=0.00006)




###############################################################################


################################## LIGHTGBM ##################################

tuned_paramaters = [{'n_estimators':[150,350,650,850,1000],
                     'learning_rate':[0.001,0.01,0.1,0.2,0.3],
                     'num_leaves':[2,3,5,7,9]}]

model_lgb = lgb.LGBMRegressor()
regrLGB = GridSearchCV(model_lgb, tuned_paramaters, cv = 5, scoring='neg_mean_squared_error')
regrLGB.fit(train, y_train)
print(np.sqrt(-regrLGB.best_score_))
regrLGB.best_score_
regrLGB.best_estimator_
regrLGB.best_params_

ypredX = regrLGB.predict(test)
ypredXc = ypredX ** 3
ind = [i for i in range(1, 1336)]
df = pd.DataFrame({'id':ind, 'views':ypredXc})
df.to_csv('submodLGB3.csv', index=False) 

lgb_new = lgb.LGBMRegressor(learning_rate=0.01, n_estimators=850, num_leaves=7)


#################################################################
#tuned_paramaters = [{'n_estimators':[650,850,1000,2000],
#                     'learning_rate':[0.01,0.1,0.05,0.3],
#                     'max_depth':[5,10,15,20]}]
#
#GBoost = GradientBoostingRegressor()
#regrGB = GridSearchCV(GBoost, tuned_paramaters, cv = 5, scoring='neg_mean_squared_error')
#regrGB.fit(train, y_train)
#regrGB.best_score_
#regrGB.best_estimator_
#regrGB.best_params_

########################### Random Forest Regression ##########################
from sklearn.ensemble import RandomForestRegressor

tuned_paramaters = [{'n_estimators':[350,650,850,1000],
                     'max_depth':[15,20,30,40,50]}]

randReg = RandomForestRegressor()

regrRF = GridSearchCV(randReg, tuned_paramaters, cv = 5, scoring='neg_mean_squared_error')
regrRF.fit(train, y_train)
print(np.sqrt(-regrRF.best_score_))
regrRF.best_params_
ypredX = regrRF.predict(test)
ypredXc = ypredX ** 3
df = pd.DataFrame({'views':ypredXc})
df.to_csv('submodRF.csv') 

randReg = RandomForestRegressor(max_depth=30, n_estimators=650)
randReg.fit(train, y_train)
print(np.sqrt(-randReg.best_score_))
coef = randReg.feature_importances_
sorted(coef)
d = pd.DataFrame({'Columns':train.columns, 'Imp':coef})
d.sort_values(['Imp'], ascending=False)
d.plot()


randReg = RandomForestRegressor(n_estimators=850, max_depth=15)
###############################################################################

###############################################################################
###############################################################################
# Stacking Regressor
from mlxtend.regressor import StackingCVRegressor
stack_gen = StackingCVRegressor(regressors=(lgb_new, randReg),
                                meta_regressor=lgb_new,
                                use_features_in_secondary=True)

stack_gen.fit(train, y_train)
stack_gen.score

ypredX = stack_gen.predict(test)
ypredXc = ypredX ** 3
ind = [i for i in range(1, 1336)]
df = pd.DataFrame({'id':ind, 'views':ypredXc})
df.to_csv('submodSCV.csv', index=False) 

###############################################################################

from sklearn.kernel_ridge import KernelRidge
knr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

###############################################################################
lasso_final = regrL
ridge_final = regr
elastic_final = regrE
lgb_final = regrLGB
rand_final = randReg.fit(train, y_train)
xgb_final = xgboost.fit(train, y_train)
stack_final = stack_gen.fit(train, y_train)
knr_final = knr.fit(train, y_train)

def blend_models_predict(X):
    return ((0.1 * elastic_final.predict(X)) + \
            (0.05 * lasso_final.predict(X)) + \
            (0.1 * ridge_final.predict(X)) + \
            (0.1 * knr_final.predict(X)) + \
            (0.15 * xgb_final.predict(X)) + \
            (0.1 * lgb_final.predict(X)) + \
            (0.3 * stack_final.predict(np.array(X))))

ypred = blend_models_predict(test)

ypredXc = ypred ** 3
ind = [i for i in range(1, 1336)]
df = pd.DataFrame({'id':ind, 'views':ypredXc})
df.to_csv('submodBlend.csv', index=False) 

###########################################################################################################
###########################################################################################################
################################### ----------- END ------------ ##########################################
###########################################################################################################
###########################################################################################################

