
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import shuffle


# Set random seed
SEED = 98
np.random.seed(SEED)

# Define variables
INPUT_VARS = ['age_cal','weight','height', 'cuffed']
OUTPUT_VAR = 'size'

# Load data
df = pickle.load(open('data_size', 'rb'))
df = shuffle(df, random_state=SEED)

x = df.loc[:, INPUT_VARS].values.astype(float)
y = df[[OUTPUT_VAR]].values.flatten().astype(float)

# Split data into train and test sets
nsamp = len(y)
ntest = int(nsamp * 0.2)
ntrain = nsamp - ntest
x_test = x[-ntest:, :]
y_test = y[-ntest:]
x_train = x[:ntrain, :]
y_train = y[:ntrain]


# Set up the hyperparameter tuning
param_dict = {
                'learning_rate': [ 0.01, 0.03, 0.05, 0.07],
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
gs.best_estimator_.get_booster().save_model('best_size_model.model')