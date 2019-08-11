# -*- coding: utf-8 -*-
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np

# Download dataset
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# download dataset   
fetch_housing_data()
print("loading Data ...")
# load data
housing = load_housing_data()
# print info
print("Done: print head:")
print(housing.head())
print("*** infos: ")
print(housing.info())
# get more information on "ocean_proximity" attribute
print("*** Ocean proximity attribute:")
print(housing["ocean_proximity"].value_counts())
print("*** Describe global dataset : ")
print(housing.describe())

# Plot histograms
import matplotlib.pyplot as plt
#housing.hist(bins=50, figsize=(30,15))
#plt.show()

# split data into training / testing sets
# - method 1 basic splitting: keep 20% of dataset as test set
#from sklearn.model_selection import train_test_split
#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# - method 2  use stratified sampling by median income categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3., 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

#housing["income_cat"].hist()
#plt.show()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    train_set = housing.loc[train_index]
    test_set = housing.loc[test_index]
# remove income_cat from dataset
for set_ in (train_set, test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
print("train set length: ",len(train_set))
print("test set length: ",len(test_set))

# further visualize data
housing = train_set.copy()

#housing.plot(kind="scatter", x="longitude", y="latitude")
#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#             s=housing["population"]/100., label="population", figsize=(10,7),
#             c="median_house_value",
#             cmap=plt.get_cmap("jet"), colorbar=True)
#plt.legend()
#plt.show()

# look for correlations
corr_matrix = housing.corr()
# - print some correlation values for median_house_value
mhv_corr = corr_matrix["median_house_value"].sort_values(ascending=False)
print("****** Correlations: median_house_value")
print(mhv_corr)
# - plot correlations
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income",
              "total_rooms", "housing_median_age"]
#scatter_matrix(housing[attributes],figsize=(12,8))
#plt.show()

# clean and normalize data
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()
# build pipeline for numerical attributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# split dataset into numerical and categorical datasets
housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
housing_cat = housing[cat_attribs]
# numerical pipeline
num_pipeline = Pipeline([
                        ('imputer',SimpleImputer(strategy="median")), # fill missing values
                        ('std_scaler', StandardScaler()), # scale num variables between 0 and 1
                        ])

# complete pipeline
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
        ])

# build final pre processed data
housing_prepared = full_pipeline.fit_transform(housing)

## Train a first model !!
#from sklearn.linear_model import LinearRegression
#
#lin_reg = LinearRegression()
#lin_reg.fit(housing_prepared, housing_labels)
#
## check some values
#some_data = housing.iloc[:5]
#some_labels = housing_labels.iloc[:5]
#some_data_prepared = full_pipeline.transform(some_data)
#print("Predictions:",lin_reg.predict(some_data_prepared))
#print("Labels:",list(some_labels))
#
## measure error
#from sklearn.model_selection import cross_val_score
#
#from sklearn.metrics import mean_squared_error
#housing_predictions = lin_reg.predict(housing_prepared)
#lin_mse = mean_squared_error(housing_labels,housing_predictions)
#lin_rmse = np.sqrt(lin_mse)
#
#print("LinearRegression RMSError=",lin_rmse)
#
## test decision tree model
#from sklearn.tree import DecisionTreeRegressor
#
#tree_reg = DecisionTreeRegressor()
#tree_reg.fit(housing_prepared, housing_labels)
#housing_predictions = tree_reg.predict(housing_prepared)
#tree_mse = mean_squared_error(housing_labels,housing_predictions)
#tree_rmse = np.sqrt(tree_mse)
#print("DecisionTreeRegressor RMSError=",tree_rmse)
#
#scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                         scoring="neg_mean_squared_error",cv=10)
#
#rmse_scores = np.sqrt(-scores)
#print("Mean CV error:",np.mean(rmse_scores))
#print("StdDev CV error:",np.std(rmse_scores))
#
# test random forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

forest_reg = RandomForestRegressor(n_estimators=50,max_features=10)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels,housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("RandomForestRegressor RMSError=",forest_rmse)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error",cv=10)

rmse_scores = np.sqrt(-scores)
print("Mean CV error:",np.mean(rmse_scores))
print("StdDev CV error:",np.std(rmse_scores))

## fine tuning
#
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestRegressor
#
#param_grid = [{'n_estimators':[3,10,30,50],
#               'max_features':[2,4,6,8,10]}]
#
#forest_reg = RandomForestRegressor()
#
#grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                           scoring='neg_mean_squared_error',
#                           return_train_score=True)
#
#grid_search.fit(housing_prepared, housing_labels)
#
#print("Best params: ",grid_search.best_params_)


