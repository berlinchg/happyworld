# -*- coding: utf-8 -*-
"""
HAPPY WORLD
This script analyzes the relationship between a country's socioeconomic
factors and the happiness of its population.

Author:         Berlin Cheng
Create Date:    2020-05-09 
Last Updated:   2020-05-10

@author: bcoy
"""
# =============================================================================
# ML TECHNIQUES:
    # (X) Linear regression
    # (X) Random forest
    # (X) Clustering
    # ( ) Classification (?)

# VISUALIZATIONS:
    # Map
    # 1 viz per technique
    # Happiness ranking
       
# WRITE UP:
    # Intro / connection to me
    # Findings
    # Room for improvement
    # Link to my bio
    
# =============================================================================
import pandas as pd
import numpy as np
import os 
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor

import eli5 
from eli5.sklearn import PermutationImportance

from sklearn import tree
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/bcoy/Anaconda3/Library/bin/graphviz'

from sklearn import datasets
from sklearn.cluster import KMeans, MeanShift, DBSCAN, Birch
from sklearn import metrics

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots 


''''''''''''''''''''''''''''''''' Import Data '''''''''''''''''''''''''''''''''

# Import the World Happiness Rankings dataset
filepath2 = '/Users/bcoy/Downloads/data-science-project/world-happiness-report-2020/'

os.chdir(filepath2)
os.getcwd()

data_whr = pd.read_csv('WHR20_DataForFigure2.1.csv')


# Import the World Factbook Dataset
filepath3 = '/Users/bcoy/Downloads/data-science-project/countries-of-the-world/'

os.chdir(filepath3)
os.getcwd()

data_fact = pd.read_csv('countries of the world (2).csv')

data_fact.head()



''''''''''''''''''''''''' Clean datasets and merge '''''''''''''''''''''''''''

# Check that data types align for the key merging field
data_whr['Country'] = data_whr['Country name'].astype(str)
data_fact['Country'] = data_fact['Country'].astype(str)

print(data_whr['Country name'].dtypes)
print(data_fact['Country'].dtypes)

# Noticed "Country" had white spaces, which was causing issue with the merge
# Strip white spaces before merge to solve this issue
print(data_fact['Country'][0])
print(data_whr['Country'][0])

data_fact['Country'] = [x.strip() for x in data_fact.Country]
data_whr['Country name'] = [x.strip() for x in data_whr['Country name']]


# Merge the World Happiness Ranking and World Factbook datasets
data = pd.merge(data_whr[['Country name','Ladder score']]
                ,data_fact
                ,left_on=str('Country name'),right_on=str('Country')
                )
# 137 records 


# Check features of interest for null values
feature_names = ['Phones (per 1000)'
                ,'Literacy (%)'
                ,'Net migration'
                ,'GDP ($ per capita)'
                ,'Pop. Density (per sq. mi.)'
                ,'Infant mortality (per 1000 births)']

for fname in feature_names:
    bool_series = pd.isnull(data[fname])
    data[bool_series]
    i = len(data.loc[data[fname].isnull()])
    if i >= 1:
        print('\nThe feature',fname,'has',i,'null values.')
        print('The null values are for:\n',data['Country'].loc[data[fname].isnull()])


# Drop rows with missing values 
# (May consider imputing if nulls are more prevalent in dataset)
data.shape
# (137, 22)

data = data.drop([41,34,79],axis=0)

data.shape
# (134, 22)



''''''''''''''''''''''''''''' Linear Regression '''''''''''''''''''''''''''''

# Define function for scatterplot
def scatter(data,x,y):
    plt.scatter(data[x],data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

scatter(data,'Ladder score','GDP ($ per capita)')

# Loop through features and print scatterplot of relationship to ladder score
for fname in feature_names:
    print('Scatterplot of Ladder Score vs.',fname)
    scatter(data,fname,'Ladder score')

# Loop through features and run linear regression, fit the model, 
# visualize, and then calculate the normalized RMSE
for fname in feature_names:
    # Create X and Y variables
    X = np.array(data[fname]).reshape(-1,1)
    y = np.array(data['Ladder score'])
    
    # Run linear regression model
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=2/3, random_state=0)
    
    # Fit a model
    regressor = linear_model.LinearRegression()
    regressor.fit(train_X,train_y)
    
    pred = regressor.predict(test_X)


    print("\nLinear Regression, Ladder Score vs.",fname)
    
    # Visualize - the testing dataset
    plt.scatter(test_X, test_y, color= 'dodgerblue')
    plt.plot(train_X, regressor.predict(train_X), color = 'red')
    plt.title ("Linear Regression of Testing Dataset")
    plt.xlabel(fname)
    plt.ylabel("Ladder score")
    plt.show()
    
    # Calculate normalized RMSE to check accuracy of linear regression model
    rmse = np.sqrt(metrics.mean_squared_error(test_y, pred))
    nrmse = rmse / (test_y.max() - test_y.min())
    print('\nNormalized Root Mean Squared Error for',fname,':',nrmse)
    
    
    

''''''''''''''''''''''''''' Decision Tree Modeling '''''''''''''''''''''''''''

# Select prediction target
y = data['Ladder score']
print(y)

# Choose "Features"
list(data.columns)

X = data[[
        'Phones (per 1000)'
        ,'Literacy (%)'
        ,'Net migration'
        ,'GDP ($ per capita)'
        ,'Pop. Density (per sq. mi.)'
        ,'Infant mortality (per 1000 births)'
        ]]

X.describe()
X.head()


# Split dataset into test and train
train_X, test_X, train_y, test_y = train_test_split(X,y,random_state = 0)

# Define model
tree_model = DecisionTreeRegressor()

# Fit model
tree_model.fit(train_X,train_y)

# Get predicted scores on validation data
test_predictions = tree_model.predict(test_X)
print(test_predictions)

print('test_y output:',test_y)
print('test_predictions output:',test_predictions)

print('Mean absolute error for tree: \t'
      ,mean_absolute_error(test_y,test_predictions))
# Mean absolute error for tree:    0.6282294287647059


# Experimenting with other decision tree models below

# Define function to calculate mean absolute error
def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X,train_y)
    preds_val = model.predict(test_X)
    mae = mean_absolute_error(test_y, preds_val)
    return(mae)

# Run MAE of a range of max_leaf_nodes to determine optimal number
for max_leaf_nodes in [5,10,20,30,40,50]:
    my_mae = get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y)
    print('Max leaf nodes: ',max_leaf_nodes,'\tMean Average Error: ',my_mae,'\t')
# 20 leaf nodes is most optimal

# Try Random Forest model
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X,train_y)
melb_preds = forest_model.predict(test_X)
print('Mean absolute error for forest:\t',mean_absolute_error(test_y,melb_preds))
# Mean absolute error for forest:  0.48825764580294134

# Try XGBoost model
xgb_model = XGBRegressor()
xgb_model.fit(train_X, train_y)

predictions = xgb_model.predict(test_X)
print('Mean absolute error for XGBoost: '
      ,mean_absolute_error(predictions,test_y))
#Mean absolute error for XGBoost:  0.5056874120985754

# Parameter tuning of XGBoost
xgb_model_2 = XGBRegressor(n_estimators=500)
#xgb_model_2 = XGBRegressor(n_estimators=500, learning_rate=0.05)
xgb_model_2.fit(train_X,train_y
             ,early_stopping_rounds=5
#             ,etest_set=[(test_X,test_y)]
             ,verbose=False
             )

predictions = xgb_model_2.predict(test_X)
print('Mean absolute error for XGBoost: '
      ,mean_absolute_error(predictions, test_y))
# Mean absolute error for XGBoost:  0.5066777046780575
# Mean absolute error for XGBoost:  0.5186772767009169 - with learning rate



''''''''''''''''''''''''''' Develop explainability '''''''''''''''''''''''''''

# Permutation Importance - "What variables most affect happiness predictions?"
perm = PermutationImportance(forest_model, random_state=1).fit(test_X,test_y)

print('Features and their importance')
list(X.columns)
list(perm.feature_importances_)

eli5.explain_weights(perm,feature_names = test_X.columns.tolist())
# this shows more details of the specific features, their STDEV, etc.

# Partial Dependence Plot - Forest Model
pdp_goals = pdp.pdp_isolate(model=forest_model, dataset=test_X
                            , model_features=feature_names
                            , feature='GDP ($ per capita)')
pdp.pdp_plot(pdp_goals, 'GDP per capita')
plt.show()

# Loop through features and print partial dependency plot for forest model
for fname in feature_names:
    pdp_goals = pdp.pdp_isolate(model=forest_model
                                , dataset=test_X
                                , model_features=feature_names
                                , feature=fname)
    pdp.pdp_plot(pdp_goals, fname)


# Partial Dependence Plot - XGB Model
if isinstance(test_X,pd.DataFrame):
    print("is dataframe")
    
print(test_X)
print(feature_names)
    
feature_names = [i for i in X.columns]
print(feature_names)
pdp_goals = pdp.pdp_isolate(model=xgb_model
                            , dataset=test_X
                            , model_features=feature_names
                            , feature='GDP ($ per capita)')
pdp.pdp_plot(pdp_goals, 'GDP per capita')
plt.show()

# Loop through features and print partial dependency plot for xgboost model
for fname in feature_names:
    pdp_goals = pdp.pdp_isolate(model=xgb_model
                                , dataset=test_X
                                , model_features=feature_names
                                , feature=fname)
    pdp.pdp_plot(pdp_goals, fname)
    


# SHAP Values - "How did this model work for a given country's happiness prediction?" 
# - Why was it different than the prediction for baseline values?
row_to_show = 5
data_for_prediction = test_X.iloc[row_to_show]
data_for_prediction_array = data_for_prediction.values.reshape(1,-1)
print(data_for_prediction_array)

print(train_y.dtypes)





'''''''''''''''''''''''''''' Clustering model '''''''''''''''''''''''''''''   

# Identify sum of square distances
ssd= []
N = range(1,20)
for n in N:
    km = KMeans(n_clusters=n)
    km = km.fit(X)
    ssd.append(km.inertia_)
    
len(ssd)

# Identify optimal k based on elbow graph
plt.plot(N, ssd, 'bx-')
plt.xlabel('N')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()
# According to elbow curve, n_clusters = 4 is optimal

# Run KMeans
kmeans = KMeans(n_clusters=4)

kmeans.fit(X)
kmeans.cluster_centers_.shape
# KMeans created 3 clusters with 6 features

data['Cluster'] = kmeans.fit_predict(X)

data[['Ladder score','Cluster']]



''''''''''' Post-Processing / Export results for visualization '''''''''''

# Import longitude + latitude info for visualization
filepath3 = '/Users/bcoy/Downloads/data-science-project/counties-geographic-coordinates/'

data_geo = pd.read_csv('countries.csv')
data_geo.head()

# Merge onto final dataset
data_output = pd.merge(data,data_whr[['Country name','Regional indicator']]
                        ,left_on='Country name',right_on='Country name')
data_output = data_output.merge(data_geo,left_on='Country name'
                                ,right_on='name')


# Write output results to CSV file
filepath_out = '/Users/bcoy/Downloads/data-science-project/'

os.chdir(filepath_out)
os.getcwd()

data_output.to_csv('data_output.csv')