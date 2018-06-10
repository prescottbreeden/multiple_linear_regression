# Multiple Linear Regression
'''
# Key Assumptions
=================
1. Linearity
2. Homoscedasticity
3. Multivariate normailty
4. Independence of errors
5. Lack of multicollinearity

# 5 Methods of Building Models
==============================
1. All-in
    - prior knowledge; or
    - you have to; or
    - preparing for backward elimination
    
2. Backward Elimination ** using this one **
    step 1: select a significance level to stay in the model (e.g. SL = .05)
    step 2: fit the full model with all possible predictors
    step 3: consider the predictor with the highest p-value, if P > SL, fo to step 4, otherwise go to FIN
    step 4: Remove the predictor
    step 5: Fit model without this variable*
    
3. Forward Selection
    step 1: Select a SL to enter the model (eg. .05)
    step 2: Fit all simple regression models y ~ Xn Select the one with the lowest P-value
    step 3: Keep this variable and fit all possible models with one extra predictor added to the ones you already have
    step 4: Consider the predictor with the lowest P-value. If P < SL, go to step 3, otherwise go to FIN
    step 5: Keep the previous model
        
4. Bidirectional Elimination
    step 1: Select a p-value level to enter and to stay in the model (SLEnter = .05, SLStay = 0.05)
    step 2: Perform the next step of Forward Seleciton (new variables must have: P < SLEnter to enter)
    step 3: Perform ALL steps of Backward Elimination (old variables must have P < SLStay to stay)
    step 4: No new variables can enter and no old variables can exit
    step 5: model is ready

5. Score Comparison
    step 1: Select a criterion of goodness of fit (e.g. Akaike criterion)
    step 2: Construct all possible regression Models 2^n-1 total combinations
    step 3: Select the one with the best criterion
    step 4: Your model is ready
    
    example: 10 columns means 1,023 models

'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
#X_opt = X[:, [0,1,2,3,4,5]]
#X_opt = X[:, [0,1,3,4,5]]
#X_opt = X[:, [0,3,4,5]]
#X_opt = X[:, [0,3,5]]
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# RESULT: R&D spending extremely predictive of profits


#
