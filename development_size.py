import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import shuffle

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Set random seed
SEED = 98
np.random.seed(SEED)

# Load the dataset
df = pd.read_csv('dataset/df_size_2308.csv', index_col = 0)

# Sort the data in ascending order based on the 'opdate' column (most recent first)
df.sort_values(by=['opdate'], ascending=True, inplace=True)

# Extract features (x) and target (y) values from the dataFrame
x = df[['age_cal', 'sex', 'weight', 'height', 'cuffed']].astype(float).values
y = df['airway_tube_size'].astype(float).values
c = df['opid'].values

# Separate the data for cuffed and uncuffed endotracheal tubes
## Cuffed endotrahceal tube
x_cuffed = x[x[:, 4] == 1][:, :4]
y_cuffed = y[x[:, 4] == 1]

# Determine the size of the test and train sets for cuffed data
nsamp = len(y_cuffed)
ntest = int(nsamp * 0.2)
ntrain = nsamp - ntest

# Split the cuffed data into training and test sets
x_cuff_test = x_cuffed[-ntest:, :]
y_cuff_test = y_cuffed[-ntest:]
x_cuff_train = x_cuffed[:ntrain, :]
y_cuff_train = y_cuffed[:ntrain]

# Impute missing values in the cuffed data using the multiple imputation method
imp = IterativeImputer().fit(x_cuff_train)
x_cuff_train = imp.transform(x_cuff_train)
x_cuff_test = imp.transform(x_cuff_test)

# Exclude the 'sex' feature and concatenate 'age', 'weight', and 'height' features into a new feature set (BorutaSHAP result)
x_cuff_train = np.concatenate((x_cuff_train[:,0:1], x_cuff_train[:,2:4]),axis=-1)
x_cuff_test = np.concatenate((x_cuff_test[:,0:1], x_cuff_test[:,2:4]),axis=-1)

## Uncuffed endotracheal tube
x_uncuffed = x[x[:, 4] == 0][:, :4]
y_uncuffed = y[x[:, 4] == 0]

# Determine the size of the test and train sets for uncuffed data
nsamp = len(y_uncuffed)
ntest = int(nsamp * 0.2)
ntrain = nsamp - ntest

# Split the uncuffed data into training and test sets
x_uncuff_test = x_uncuffed[-ntest:, :]
y_uncuff_test = y_uncuffed[-ntest:]
x_uncuff_train = x_uncuffed[:ntrain, :]
y_uncuff_train = y_uncuffed[:ntrain]

# Impute missing values in the uncuffed data using the multiple imputation method
imp = IterativeImputer().fit(x_uncuff_train)
x_uncuff_train = imp.transform(x_uncuff_train)
x_uncuff_test = imp.transform(x_uncuff_test)

# Exclude the 'sex' feature and concatenate 'age', 'weight', and 'height' features into a new feature set (BorutaSHAP result)
x_uncuff_train = np.concatenate((x_uncuff_train[:,0:1], x_uncuff_train[:,2:4]),axis=-1)
x_uncuff_test = np.concatenate((x_uncuff_test[:,0:1], x_uncuff_test[:,2:4]),axis=-1)

# Print the shapes of the cuffed and uncuffed training and test sets
print(f'x_cuff_train: {(x_cuff_train).shape}, x_cuff_test: {x_cuff_test.shape}')
print(f'x_uncuff_train: {(x_uncuff_train).shape}, x_uncuff_test: {x_uncuff_test.shape}')


## GBRT model for uncuffed endotracheal tube
# Set up the hyperparameter tuning
param_dict = {
                'learning_rate': [ 0.01, 0.03, 0.05, 0.07],
                'max_depth': [3, 4, 5, 7],
                'n_estimators': [25, 50, 75, 100, 300],
                'subsample': [0.5, 0.8, 1], 
                'colsample_bytree': [0.5, 0.8, 1], 
                'gamma': [0.3, 0.5, 0.7, 0.9]
            }
nfold = 10
gs = GridSearchCV(estimator=xgb.sklearn.XGBRegressor(),
                n_jobs=-1,
                verbose=2,
                param_grid=param_dict, cv=nfold)

# Train the model
gs.fit(x_uncuff_train, y_uncuff_train)

# Get and print the best hyperparameters and score
print("Gradient-boosted regression tree model for predicting uncuffed endotracheal tube size")
print("========= found hyperparameter =========")
print(gs.best_params_)
print(gs.best_score_)
print("========================================")

# Save the best model
#gs.best_estimator_.get_booster().save_model('gbrt_size_uncuffed.json')
gbrt_uncuff = gs.best_estimator_.get_booster()
gbrt_uncuff.save_model('gbrt_size_uncuffed.model')

y_pred = gbrt_uncuff.predict(xgb.DMatrix(x_uncuff_test)).flatten()  # remove xgb.DMatrix if error is raised
y_pred = np.round(y_pred * 2) / 2

## GBRT model for cuffed endotracheal tube
# Set up the hyperparameter tuning
param_dict = {
                'learning_rate': [ 0.01, 0.03, 0.05, 0.07],
                'max_depth': [3, 4, 5, 7],
                'n_estimators': [25, 50, 75, 100, 300],
                'subsample': [0.5, 0.8, 1], 
                'colsample_bytree': [0.5, 0.8, 1], 
                'gamma': [0.3, 0.5, 0.7, 0.9]
            }
nfold = 10
gs = GridSearchCV(estimator=xgb.sklearn.XGBRegressor(),
                n_jobs=-1,
                verbose=2,
                param_grid=param_dict, cv=nfold)

# Train the model
gs.fit(x_cuff_train, y_cuff_train)

# Get and print the best hyperparameters and score
print("Gradient-boosted regression tree model for predicting cuffed endotracheal tube size")
print("========= found hyperparameter =========")
print(gs.best_params_)
print(gs.best_score_)
print("========================================")

# Save the best model
#gs.best_estimator_.get_booster().save_model('gbrt_size_cuffed.json')
gbrt_cuff = gs.best_estimator_.get_booster()
gbrt_cuff.save_model('gbrt_size_cuffed.model')

y_pred = gbrt_cuff.predict(xgb.DMatrix(x_cuff_test)).flatten() # remove xgb.DMatrix if error is raised
y_pred = np.round(y_pred * 2) / 2
