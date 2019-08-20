from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import pandas as pd
import seaborn as sns
import numpy as np

from tensorflow import keras
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# get data
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print("Dataset downloaded in :"+dataset_path)

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']


raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

def visualize_dataset(dataset):
    """
    Explore and plot dataset
    """
    print("***** Visualize dataset *****")
    print(dataset.info())
    print(dataset.describe())
    # Plot histograms
    print(" - Histograms plots")
    dataset.hist(bins=50, figsize=(30,15))
    corr_matrix = dataset.corr()
    # Plot correlations
    mpg_corr = corr_matrix["MPG"].sort_values(ascending=False)
    print(" - Correlations: MPG")
    print(mpg_corr)
    attributes = column_names
    pd.plotting.scatter_matrix(dataset[attributes],figsize=(12,8))
    plt.show()
    
def prepare_data(dataset):
    dataset = dataset.dropna()
    origin = dataset.pop('Origin')

    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0
    dataset.tail()

    train_x = dataset.sample(frac=0.8,random_state=0)
    test_x = dataset.drop(train_x.index)
    
    train_y = train_x.pop('MPG')
    test_y = test_x.pop('MPG')
    return train_x,train_y,test_x,test_y

def prepare_data_v2(dataset):
    """
    Prepare data for ML algo:
    - Encode categorical attributes
    - Scale numerical attributes
    """
    print("***** Preparing data *****")
    data_x = dataset.drop("MPG", axis=1)
    data_y = dataset["MPG"].copy()
    # split dataset into numerical and categorical datasets
    x_num = data_x.drop("Origin", axis=1)
    num_attribs = list(x_num)
    cat_attribs = ["Origin"]
    x_cat = data_x[cat_attribs]
    print(" - Numerical attributes: "+str(num_attribs))
    print(" - Categorical attributes: "+str(cat_attribs))
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

    
#visualize_dataset(raw_dataset)
#train_x,train_y,test_x,test_y = prepare_data(raw_dataset)
train_data = raw_dataset.sample(frac=0.8,random_state=0)
test_data = raw_dataset.drop(train_data.index)

train_x,train_y = prepare_data_v2(train_data)
test_x,test_y = prepare_data_v2(test_data)

#model = tune_model(train_x, train_y, model="SVR", tuning_method="Random")
#print(model)

# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(max_features=7,n_estimators=250)
# model.fit(train_x,train_y)

from sklearn.svm import SVR
model = SVR(C=80.,gamma=0.2)
model.fit(train_x, train_y)

# valid model
predicted_y = model.predict(test_x)

lin_mse = mean_squared_error(test_y,predicted_y)
lin_rmse = np.sqrt(lin_mse)
print("RMS Error = "+str(lin_rmse))
scatter_plot_with_correlation_line(test_y,predicted_y)


