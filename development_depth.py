
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle


# Set random seed
SEED = 98
np.random.seed(SEED)

# Define variables
INPUT_VARS = ['age_cal','sex', 'weight','height']
OUTPUT_VAR = 'depth'

# Load data
df = pd.read_csv('data_depth.csv')
df = shuffle(df, random_state=SEED)

x = df.loc[:, INPUT_VARS].values.astype(float)
y = df[[OUTPUT_VAR]].values.flatten().astype(float)

# spliiting into train and test sets
nsamp = len(y)
ntest = int(nsamp * 0.2)
ntrain = nsamp - ntest
x_test = x[-ntest:, :]
y_test = y[-ntest:]
x_train = x[:ntrain, :]
y_train = y[:ntrain]

# Input variables after feature selection: age, sex, weight, height
x_train = x_train[:,0:4]
x_test = x_test[:,0:4]


# Set up the hyperparameter tuning
param_dict = {
                'learning_rate': [0.01, 0.05, 0.1], 
                'max_depth': [3, 4, 5, 7],
                'n_estimators': [25, 50, 75, 100, 300],
                'subsample': [0.5, 0.8, 1], 
                'colsample_bytree': [0.5, 0.8, 1], 
                'gamma': [0.3, 0.5, 0.7, 0.9],
                'scale_pos_weight': [1, 10, 30, 100]
            }
nfold = 10
gs = GridSearchCV(estimator=xgb.sklearn.XGBRegressor(),
                n_jobs=-1,
                verbose=2,
                param_grid=param_dict, cv=nfold)

# Train the model
gs.fit(x_train, y_train)

# Get and print the best hyperparameters and score
print("========= found hyperparameter =========")
print(gs.best_params_)
print(gs.best_score_)
print("========================================")

# Save the best model
gs.best_estimator_.get_booster().save_model('best_depth_model.model')

