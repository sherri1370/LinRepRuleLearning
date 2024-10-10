import numpy as np
import pandas as pd



class BootstrapSplitter:

    def __init__(self, reps, train_size, replace = True, random_state=None):
        self.reps = reps
        self.train_size = train_size
        self.RNG = np.random.default_rng(random_state)
        self.replace = replace

    def get_n_splits(self):
        return self.reps

    def split(self, x, y=None, groups=None):
        for _ in range(self.reps):
            train_idx = self.RNG.choice(np.arange(len(x)), size=self.train_size, replace=self.replace)
            test_idx = np.setdiff1d(np.arange(len(x)), train_idx)
            np.random.shuffle(test_idx)
            yield train_idx, test_idx



class CallData:

    def __init__(self, filepath = 'datasets/'):

        self.filepath = filepath
        self.feature_names_dict = {
            'california_housing_price': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'],
            'used_car_price': ['count', 'km', 'year', 'powerPS'],
            'red_wine': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
            'diabetes':['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'],
            'make_friedman1': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
            'synthetic1': ['x1','x2'],
            'breast_cancer': ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                            'mean smoothness', 'mean compactness', 'mean concavity',
                            'mean concave points', 'mean symmetry', 'mean fractal dimension',
                            'radius error', 'texture error', 'perimeter error', 'area error',
                            'smoothness error', 'compactness error', 'concavity error',
                            'concave points error', 'symmetry error', 'fractal dimension error',
                            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                            'worst smoothness', 'worst compactness', 'worst concavity',
                            'worst concave points', 'worst symmetry', 'worst fractal dimension'],
            'iris': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
            'banknote': ['variance', 'skewness', 'curtosis', 'entropy'],
            'magic04': ['fLen1t-1',	'fWidt-1', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Lon1', 'fM3Trans', 'fAlp-1a', 'fDist'],
            'voice': ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid', 
                      'meanfun', 'minfun', 'maxfun',
                      'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx'],
            'synthetic': ['x1', 'x2']
        }
        self.target_name_dict = {
            'california_housing_price': 'MedHouseVal',
            'used_car_price': 'avgPrice',
            'red_wine': 'quality',
            'diabetes': 'target',
            'make_friedman1': 'y',
            'synthetic1': 'y',
            'breast_cancer': 'y',
            'iris': 'y',
            'banknote': 'class',
            'magic04': 'class',
            'voice': 'label',
            'synthetic': 'y'
        }
        self.learning_type_dict = {
            'california_housing_price': 'linreg',
            'used_car_price': 'linreg',
            'red_wine': 'linreg',
            'diabetes': 'linreg',
            'make_friedman1': 'linreg',
            'synthetic1': 'linreg',
            'breast_cancer': 'logreg',
            'iris': 'logreg',
            'banknote': 'logreg',
            'magic04': 'logreg',
            'voice': 'logreg',
            'synthetic': 'logreg'
        }

    def standardize(self, x):
        x = (x - x.mean(axis=0))/(x.std(axis=0))
        return x

    def call(self, dataset_name = 'california_housing_price', data_format = 'arr'):
        """
        data_format: 'arr' (numpy array) or 'df' (pandas dataframe)
        """
        filename = dataset_name+'.csv'
        data = pd.read_csv(f'{self.filepath}{filename}')
        self.learning_type = self.learning_type_dict[dataset_name]
        X_df = data[self.feature_names_dict[dataset_name]]
        y_df = data[self.target_name_dict[dataset_name]]
        X_df = self.standardize(X_df)
        if self.learning_type == 'linreg':
            y_df = self.standardize(y_df)
        elif self.learning_type == 'logreg':
            y_df = (y_df>0).astype(float)
        
        if data_format == 'arr':
            self.X = X_df.to_numpy()
            self.y = y_df.to_numpy()
        else:
            self.X = X_df
            self.y = y_df
        
        return self
    