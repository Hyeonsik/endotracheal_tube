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
INPUT_VARS = ['age','sex', 'weight','height']
OUTPUT_VAR = 'airway_tube_size'

# Load data
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

# Create a DataFrame with the training features (X) from the cuffed data
X = pd.DataFrame(x_cuff_train, columns=INPUT_VARS)

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
                     y=y_cuff_train, 
                     n_trials=100, 
                     sample=False, 
                     train_or_test = 'train', 
                     normalize=True, 
                     verbose=False, 
                     random_state=SEED)

# Plotting the BorutaSHAP 
print('Boxplot chart illustrating BorutaSHAP result for cuffed endotracheal tubes')
Feature_Selector.plot(X_size=15, which_features='all')



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


# Create a DataFrame with the training features (X) from the uncuffed data
X = pd.DataFrame(x_uncuff_train, columns=INPUT_VARS)

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
                     y=y_uncuff_train, 
                     n_trials=100, 
                     sample=False, 
                     train_or_test = 'train', 
                     normalize=True, 
                     verbose=False, 
                     random_state=SEED)

# Plotting the BorutaSHAP 
print('Boxplot chart illustrating BorutaSHAP result for uncuffed endotracheal tubes')
Feature_Selector.plot(X_size=15, which_features='all')
