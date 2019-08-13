# -*- coding: utf-8 -*-
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
# scikit learn imports
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

def load_housing_data(housing_url=None, housing_path=None, overwrite=False):
    """
    Download and Extract dataset
    """
    print "***** Extracting data *****"
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join("datasets","housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    if housing_url is None:
        housing_url = HOUSING_URL
    if housing_path is None:
        housing_path = HOUSING_PATH
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    housing_csv = os.path.join(housing_path,'housing.csv')
    if not os.path.exists(housing_csv) or overwrite:
        print " - Downloading dataset from "+housing_url
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
    print " - Loading "+housing_csv
    return pd.read_csv(housing_csv)

def visualize_dataset(dataset):
    """
    Explore and plot dataset
    """
    print "***** Visualize dataset *****"
    print dataset.info()
    print dataset.describe()
    # Plot histograms
    print " - Histograms plots"
    #plt.figure()
    dataset.hist(bins=50, figsize=(30,15))
    print " - Coordinates plots"
    #plt.figure()
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=dataset["population"]/100., label="population", figsize=(10,7),
                 c="median_house_value",
                 cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    print " - Correlations plots"
    #plt.figure()
    corr_matrix = dataset.corr()
    # Plot correlations
    mhv_corr = corr_matrix["median_house_value"].sort_values(ascending=False)
    print " - Correlations: median_house_value"
    print mhv_corr
    attributes = ["median_house_value", "median_income",
                  "total_rooms", "housing_median_age"]
    pd.plotting.scatter_matrix(dataset[attributes],figsize=(12,8))
    plt.show()

def split_dataset(dataset,test_size=0.2,random_state=42, clip_dataset=False):
    """
    Split dataset into train/test sets
    """
    print "***** Splitting data in train/test sets *****"
    #  use stratified sampling by median income categories
    dataset["income_cat"] = pd.cut(dataset["median_income"],
                                   bins=[0., 1.5, 3., 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    if clip_dataset:
        dataset = dataset[dataset['median_house_value']<500000.0]
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(dataset, dataset['income_cat']):
        train_set = dataset.iloc[train_index]
        test_set = dataset.iloc[test_index]
    train_set = train_set.drop("income_cat", axis=1)
    test_set = test_set.drop("income_cat", axis=1)
    print " - Train set size: "+str(len(train_set))
    print " - Test set size: "+str(len(test_set))
    return train_set, test_set

def prepare_data(dataset):
    """
    Prepare data for ML algo:
    - Encode categorical attributes
    - Scale numerical attributes
    """
    print "***** Preparing data *****"
    data_x = dataset.drop("median_house_value", axis=1)
    data_y = dataset["median_house_value"].copy()
    # split dataset into numerical and categorical datasets
    x_num = data_x.drop("ocean_proximity", axis=1)
    num_attribs = list(x_num)
    cat_attribs = ["ocean_proximity"]
    x_cat = data_x[cat_attribs]
    print " - Numerical attributes: "+str(num_attribs)
    print " - Categorical attributes: "+str(cat_attribs)
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
    data_x_prepared = full_pipeline.fit_transform(data_x)
    return data_x_prepared, data_y


def tune_model(data_x, data_y, model="RandomForestRegressor", tuning_method="Random"):
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import GridSearchCV
    from scipy.stats import expon, reciprocal, randint
    
    if model=="RandomForestRegressor": 
        from sklearn.ensemble import RandomForestRegressor
        reg_model = RandomForestRegressor()
        param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                      {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
        param_rndm = {'n_estimators': randint(low=1, high=300),
                      'max_features': randint(low=1, high=10)}
        
    elif model=="SVR":
        from sklearn.svm import SVR
        reg_model = SVR()
        param_grid = [{'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
                      {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}]
        param_rndm = {'kernel': ['linear', 'rbf'],
                      'C': reciprocal(20, 200000),
                      'gamma': expon(scale=1.0)}

    elif model=="LinearRegression":
        from sklearn.linear_model import LinearRegression
        reg_model = LinearRegression()
        param_grid = []
        param_rndm = {}
    
    elif model=="Ridge":
        from sklearn.linear_model import Ridge
        reg_model = Ridge()
        param_grid = [{'alpha': [0.001,0.01,0.1,1.,10.,100.]}]
        param_rndm = {'alpha': expon(scale=1.0)}
        
    if tuning_method=="Random":
        rnd_search = RandomizedSearchCV(reg_model, param_distributions=param_rndm,
                                        n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                        random_state=42,verbose=2,n_jobs=4)
        rnd_search.fit(data_x, data_y)
        best_model = rnd_search.best_estimator_
    else:
        grid_search = GridSearchCV(reg_model, param_grid, cv=5,
                                   scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
        grid_search.fit(data_x, data_y)
        best_model = grid_search.best_estimator_
    return best_model

def scatter_plot_with_correlation_line(x, y, graph_filepath=None):
    '''
    http://stackoverflow.com/a/34571821/395857
    x does not have to be ordered.
    '''
    # Scatter plot
    plt.scatter(x, y)

    # Add correlation line
    axes = plt.gca()
    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.plot(X_plot, m*X_plot + b, '-')
    text = "Q2="+str(m)
    annchored_text = AnchoredText(text,prop=dict(size=12), frameon=True, loc=4)
    axes.add_artist(annchored_text)
    # Save figure
    plt.show()#savefig(graph_filepath, dpi=300, format='png', bbox_inches='tight')
    
housing = load_housing_data()
#visualize_dataset(housing)

train_data,test_data = split_dataset(housing,clip_dataset=True)
train_x,train_y = prepare_data(train_data)

model = tune_model(train_x, train_y, model="RandomForestRegressor", tuning_method="Random")
print model

# valid model
test_x,test_y = prepare_data(test_data)
predicted_y = model.predict(test_x)

lin_mse = mean_squared_error(test_y,predicted_y)
lin_rmse = np.sqrt(lin_mse)
print "RMS Error = "+str(lin_rmse)
scatter_plot_with_correlation_line(test_y,predicted_y)
    
