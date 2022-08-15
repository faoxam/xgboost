#“Application of Machine Learning in Predicting Water Saturation using Well Log Data”
#Codes:
import sys

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#input data from files
import os

#Read train CSV
#check data input in form of table
actual_series_fn = './input/' + sys.argv[1]
df_train = pd.read_csv(actual_series_fn)
df_train
print(df_train.columns.tolist())
print(df_train.columns.tolist())

X = df_train.loc[:,: sys.argv[2]]
y = df_train.loc[:,sys.argv[3]] 

# Splitting
train_X, test_X, train_y, test_y = train_test_split(X, y,
                      test_size = 0.3, random_state = 123)
#create XGBoost model
xgb.XGBClassifier().get_params()

model_xgboost = xgb.XGBClassifier(learning_rate=0.1,
                                      max_depth=5,
                                      n_estimators=5000,
                                      subsample=0.5,
                                      colsample_bytree=0.5,
                                      eval_metric='rmse',
                                      verbosity=2)

# Train and test set are converted to DMatrix objects,
# as it is required by learning API.
train_dmatrix = xgb.DMatrix(data = train_X, label = train_y)
test_dmatrix = xgb.DMatrix(data = test_X, label = test_y)
 
# Parameter dictionary specifying base learner
param = {"booster":"gblinear", "objective":"reg:linear"}
 
xgb_r = xgb.train(params = param, dtrain = train_dmatrix, num_boost_round = 10)
pred_test = xgb_r.predict(test_dmatrix)
pred_train = xgb_r.predict(train_dmatrix)

# RMSE Computation for first model
rmse = np.sqrt(MSE(test_y, pred_test))
print("RMSE test : % .3f" %(rmse))
rmse = np.sqrt(MSE(train_y, pred_train))
print("RMSE train : % .3f" %(rmse))

#show the data of the results
df = pd.DataFrame({'Actual': test_y, 'Predicted': pred_test})
print(df)
#Hyperparameters tuning of the first model
learning_rate_list = [0.02, 0.05, 0.1]
max_depth_list = [2, 3, 5]
n_estimators_list = [1000, 2000, 3000]

params_dict = {"base_estimator__learning_rate": learning_rate_list,
               "base_estimator__max_depth": max_depth_list,
               "base_estimator__n_estimators": n_estimators_list}

num_combinations = 1
for v in params_dict.values(): num_combinations *= len(v) 

print(num_combinations)
params_dict

def my_MSE(model, X, y): return MSE(y, model.predict_proba(X)[:,1])
model_xgboost_hp = GridSearchCV(estimator=xgb.XGBClassifier(subsample=0.5,
                                                                colsample_bytree=0.25,
                                                                eval_metric='RMSE',
                                                                use_label_encoder=False),
                                param_grid=params_dict,
                                cv=2,
                                scoring=my_MSE,
                                return_train_score=True,
                                verbose=1)

train_dmatrix_hp = xgb.DMatrix(data = train_X, label = train_y)
test_dmatrix_hp = xgb.DMatrix(data = test_X, label = test_y)
 
xgb_r = xgb.train(params = params_dict, dtrain = train_dmatrix_hp, num_boost_round = 10)
pred_test_hp = xgb_r.predict(test_dmatrix_hp)
pred_train_hp = xgb_r.predict(train_dmatrix_hp)

 # RMSE Computation for the tuned model
rmse = np.sqrt(MSE(test_y, pred_test_hp))
print("RMSE_hp test : % .3f" %(rmse))
rmse = np.sqrt(MSE(train_y, pred_train_hp))
print("RMSE_hp train : % .3f" %(rmse))

#Show the data of the predicted values vs actual
df_hp = pd.DataFrame({'Actual': test_y, 'Predicted': pred_test_hp})
print(df_hp)

#Find the R2 values of the tuned model and the first model.
corr_matrix = np.corrcoef(test_y,pred_test)
corr=corr_matrix[0,1]
R_sq=corr**2

print("Rsq : % .3f" %(R_sq))

corr_matrix = np.corrcoef(test_y,pred_test_hp)
corr=corr_matrix[0,1]
R_sq_hp=corr**2

print("Rsq_hp : % .3f" %(R_sq_hp))

#Export the data of predicted Sw and actual Sw
predict_file_name = './output/' + sys.argv[1].split(".")[0] +'_'+ sys.argv[2]+'_'+ sys.argv[3]+ '_predict.csv'
df.index.name='INDEX'
df['INDEX']=df.index
df = df.reset_index(drop=True)
df_hp.sort_values(by=["INDEX"], ascending=False)
df_hp.to_csv(predict_file_name, index=True)

