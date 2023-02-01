# -*- coding: utf-8 -*-
"""
Data Analytics Computing Follow Along: Random Forest for Regression and Classification
Spyder version 5.3.3
"""

# Import the required packages
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split

# We will perform regression and classification with random forests this time
# Random forests are a powerful ensemble model that combines decision trees
# For our purposes, we can think of an ensemble as a model made up of different models

# Run the lines below if you do not have the packages
# pip install pandas
# pip install sklearn


# The link is in kaggle is in the description, this is a dataset about traffic accidents in Arizona
# We will be using pandas.read_csv to import our data

accidents_df = pd.read_csv('C:/Videos/Arizona Accidents Dataset/accident.csv')

# See the shape

accidents_df.shape

# Notice that there are 80 columns, here is where a subject matter expert helps a lot
# We are taking only a subset of the data for the Random Forest models
# The variables below were already preselected

accidents_df_sub = accidents_df[['VE_TOTAL', 'PEDS', 'county_name', 
                           'man_coll_lit', 'FATALS', 'a_inter_lit',
                           'a_dow_lit']]

# VE total is the total amount of vehicles involved
# PEDS is the number of pedestrians
# County name is the count where the accident happened
# man_coll_lit is the manner of collision
# FATALS is the number of people that died in the accident
# a_inter_lit says if the accident happened in an interstate or not
# a_dow_lit is the day of week that the accident happened

# We can see the null values below

accidents_df_sub.isna().sum()

# We have no null values, we can proceed without dropping the nulls

# For the classification problem, the county name will be used as a prediction
# For the regression problem, fatalities will be used as a prediction
# All other features will be used as predictors

# See the unique values for the categorical variables

accidents_df_sub['county_name'].unique()
accidents_df_sub['man_coll_lit'].unique()
accidents_df_sub['a_inter_lit'].unique()
accidents_df_sub['a_dow_lit'].unique()

# The only variable that we will recode is county name. We will recode everything that it is not Maricopa as "other"
# We will also filter out 999 as it might have been a typo

accidents_df_sub_2 = accidents_df_sub[accidents_df_sub['county_name'] != '999']

accidents_df_sub_2['county_name'] = np.where(accidents_df_sub_2['county_name'] != 'MARICOPA', 
                                             'Other', 'Maricopa')

# We also need to recode all categorical variables to dummies
# Dummy variables allow machine learning algorithms to predict using a certain coding

accidents_df_categorical = pd.get_dummies(accidents_df_sub_2[['man_coll_lit', 
                                                            'a_inter_lit', 'a_dow_lit']])

accidents_final_df = pd.concat([accidents_df_sub_2, accidents_df_categorical], axis = 1)

# See the columns and the first rows of the dataset

print(accidents_final_df.columns)

print(accidents_final_df.head())

# For the first problem, we want to know if we can predict if an accident will be in Maricopa
# This will be predicted using all the other variables

# The drop method allows us to remove the columns that we want

X = accidents_final_df.drop(['county_name', 'man_coll_lit', 
                             'a_inter_lit', 'a_dow_lit'], axis = 1)

# We subset y as usual
y = accidents_df_sub_2['county_name']

# Start by fitting a random forest and a decision tree on all the data
# Is there an improvement?

rf_ex = RandomForestClassifier(n_estimators = 1500,
                                  random_state = 45, oob_score = True)
rf_ex.fit(X, y)

# We can get the score of the model
rf_ex.score(X, y)

# Compare it with a classification tree

ex_tree = tree.DecisionTreeClassifier(max_depth = 3)
ex_tree.fit(X, y)
ex_tree.score(X, y)

# Let the tree grow
ex_tree_full = tree.DecisionTreeClassifier()
ex_tree_full.fit(X, y)
ex_tree_full.score(X, y)


# Why use a random forest if we get the same accuracy than decision trees?
# Something that we have not discussed so far is that we usually want to train a model on data
# We can this process training
# However, we also want to predict on some other data, testing the model on new data that the model has not seen
# We will do something that we have not done so far, separating the data between training and testing sets
# This is also very useful to simulate when we have have data with the responses and data where we do not have them

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 50)

# There is quite a bit to unpack here

# The data has too many dimensions to have a appropiate visualization
# Dimension reduction and feature selection techniques such as PCA can help with this
# For a deeper discussion, see the curse of dimensionality
# The main advantages of the random forest algorithm are that it is a black box model
# It also takes advantage of taking into account a lot of trees to make a prediction
# This is the famous wisdom of the crowds
# In addition, it is called random because each tree in the forest considers a random subset of features
# Recall from the decision tree video that the tree considers all the features that we give it
# For more details about the theory behind it, boosting and bagging sections in ISL are recommended

# First import the random forest classifier again

rf_class = RandomForestClassifier(n_estimators = 1500,
                                  random_state = 45, oob_score = True)
rf_class.fit(X_train, y_train)

# We predict on the test data
rf_class_predictions = rf_class.predict(X_test)

# We can still get the feature importance
rf_class.feature_importances_

# We can have a more visual plot of the feature importances
feature_importances_class = pd.Series(rf_class.feature_importances_, 
                                index = X.columns).sort_values(ascending = True)

# We can plot the importances
feature_importances_class.plot(kind='barh')

# Because there are more than 17 predictors, we will not see if the model can hold new observations

# And just as with all predictive models that we have seen, we can get a probability instead of a prediction
class_predictions_proba = rf_class.predict_proba(X_train)

# Remember that the second one is the probability of the accident being in Maricopa County
class_predictions_proba[:, 1]

# We can get the score of the model
# Notice that we are using the test versions of X and y
rf_class.score(X_test, y_test)

# Compare it with a classification tree

class_tree = tree.DecisionTreeClassifier(max_depth = 3)
class_tree.fit(X_train, y_train)
class_tree.score(X_test, y_test)

# A 3% increase in accuracy may seem low, but this is a pruned tree, let the tree grow
class_tree_full = tree.DecisionTreeClassifier()
class_tree_full.fit(X_train, y_train)
class_tree_full.score(X_test, y_test)

# And finally, the classic Confusion Matrix
# 135 observations were true positives, 244 were true negatives
confusion_matrix(y_test, rf_class_predictions)

# And the classification report
print(classification_report(y_test, rf_class_predictions))

# Now, what if we do the same without the recoding?
# This is also referred as multiclass or multinomial classification

accidents_df_county = accidents_df_sub[accidents_df_sub['county_name'] != '999']

accidents_df_categorical_2 = pd.get_dummies(accidents_df_county[['man_coll_lit', 
                                                            'a_inter_lit', 'a_dow_lit']])

accidents_multiclass = pd.concat([accidents_df_county, accidents_df_categorical_2], axis = 1)

X_m = accidents_multiclass.drop(['county_name', 'man_coll_lit', 
                             'a_inter_lit', 'a_dow_lit'], axis = 1)

y_m = accidents_multiclass['county_name']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size = 0.30, random_state = 250)

rf_2 = RandomForestClassifier(n_estimators = 1500,
                                  random_state = 45, oob_score = True)
rf_2.fit(X_train_m, y_train_m)

# Score
rf_2.score(X_test_m, y_test_m)
rf_2.oob_score_

# Compare it with a classification tree

ex_tree_2 = tree.DecisionTreeClassifier(max_depth = 3)
ex_tree_2.fit(X_train_m, y_train_m)
ex_tree_2.score(X_test_m, y_test_m)

# Compare to a full grown
ex_tree_full_2 = tree.DecisionTreeClassifier()
ex_tree_full_2.fit(X_train_m, y_train_m)
ex_tree_full_2.score(X_train_m, y_train_m)

# Why is the random forest worse than both trees?
# More complex models are not necessarily better, keep this on mind
# Depending on the data, a simpler model can outperform a more complex one
# If the simpler model is better, keep the simpler model
# Arguably, model tuning could make the random forest outperform the decision trees
# That is outside the scope of this video

# We can see the estimators used in the ensemble
rf_class.estimators_
# And what estimator was used
rf_class.base_estimator_
# Compare the classes attributes
rf_class.classes_
rf_2.classes_
# Or the number of features
rf_class.n_features_in_
# Or the outputs
rf_class.n_outputs_
rf_2.n_outputs_

# Now, let's wrap-up with regression

# First recode county name

accidents_final_df['county_name'] = np.where(accidents_final_df['county_name'] != 'Maricopa', 
                                             0, 1)

X_reg = accidents_final_df.drop(['FATALS', 'county_name', 'man_coll_lit', 
                             'a_inter_lit', 'a_dow_lit', 'man_coll_lit_Angle',
                             'man_coll_lit_Front-to-Front', 'man_coll_lit_Front-to-Rear',
                             'man_coll_lit_Not a Collision with Motor Vehicle In-Transport',
                             'man_coll_lit_Other', 'man_coll_lit_Sideswipe - Opposite Direction',
                             'man_coll_lit_Sideswipe - Same Direction', 'man_coll_lit_Unknown',
                             'a_inter_lit_Non-Interstate', 'a_inter_lit_Unknown', 
                             'man_coll_lit_Rear-to-Side', 'man_coll_lit_Rear-to-Side'], axis = 1)

y_reg = accidents_final_df['FATALS']

# We will simulate having to use 70% of our data for testing purposes                    
            
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size = 0.70, random_state = 250)

rf_reg = RandomForestRegressor(n_estimators = 1500,
                                  random_state = 45, oob_score = True)

rf_reg.fit(X_train_reg, y_train_reg)

# Score
rf_reg.score(X_test_reg, y_test_reg)

# We can also get an out-of-bag score
rf_reg.oob_score_

# Compare it with a decision tree regressor

tree_reg = tree.DecisionTreeRegressor(max_depth = 3)
tree_reg.fit(X_train_reg, y_train_reg)
tree_reg.score(X_test_reg, y_test_reg)

# Compare to the full tree
tree_reg_full = tree.DecisionTreeRegressor()
tree_reg_full.fit(X_train_reg, y_train_reg)
tree_reg_full.score(X_test_reg, y_test_reg)

# Get the feature importance
rf_reg.feature_importances_

# We can have a more visual plot of the feature importances
feature_importances_reg = pd.Series(rf_reg.feature_importances_, 
                                index = X_train_reg.columns).sort_values(ascending = True)

# We can plot the importances
plt.title("Variable Importance")
feature_importances_reg.plot(kind='barh')


# The main advantage of the random forest is that the trees of the forest are uncorrelated
# As we can see, the random forest does not overfit the data as much
# How do we determine the number of trees?
# Using more advanced techniques such as cross-validation and hyperparameter tunning, we could improve the model