import pickle
import xgboost as xgb
import numpy as np
from sklearn.utils import shuffle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from BorutaShap import BorutaShap

# Set random seed
SEED = 98
np.random.seed(SEED)

# Define variables
INPUT_VARS = ['age_cal','sex', 'weight','height', 'cuffed']
OUTPUT_VAR = 'airway_tube_size'

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
y_test_old = y_old[-ntest:]
x_train = x[:ntrain, :]
y_train = y[:ntrain]

# Multiple imputation for missing value
imp = IterativeImputer().fit(x_train)
x_train_imputed = imp.transform(x_train)
x_test_imputed = imp.transform(x_test)
X = pd.DataFrame(x_train_imputed, columns=INPUT_VARS)

# Set up the model
xgbr = xgb.XGBRegressor(random_state=SEED, tree_method='gpu_hist')

# Set up the feature selector
Feature_Selector = BorutaShap(model=xgbr, 
                              importance_measure='shap', 
                              classification=False, 
                              percentile=100, 
                              pvalue=0.05)

# Fit the feature selector
Feature_Selector.fit(X=X, 
                     y=y_train, 
                     n_trials=100, 
                     sample=False, 
                     train_or_test = 'train', 
                     normalize=True, 
                     verbose=False, 
                     random_state=SEED)

# Plotting the BorutaSHAP 
Feature_Selector.plot(X_size=15, which_features='all')
