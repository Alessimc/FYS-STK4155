"""
This file contains general functions used throughout FYS-STK4155 project 1.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import imageio.v2 as imageio

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline

from plotting_functions import *

def FrankeFunction(x, y):
    """
    Compute the Franke function value for given x and y.

    Parameters:
    x (ndarray): Input array for x values.
    y (ndarray): Input array for y values.

    Returns:
    ndarray: Computed Franke function values.
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4

def create_X(x, y, order):
    """
    Create a design matrix for polynomial regression.

    Parameters:
    x (ndarray): Input array for x values.
    y (ndarray): Input array for y values.
    order (int): The polynomial order.

    Returns:
    ndarray: Design matrix.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((order + 1) * (order + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, order + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y ** k)

    return X

def scatterPlot(X, Y, Z, label=None, show=True, fig=None, ax=None):
    """
    Create a 3D scatter plot.

    Parameters:
    X (ndarray): X coordinates.
    Y (ndarray): Y coordinates.
    Z (ndarray): Z coordinates.
    label (str, optional): Label for the plot.
    show (bool, optional): Whether to show the plot immediately.
    fig (Figure, optional): Matplotlib figure object.
    ax (Axes3D, optional): Matplotlib 3D axes object.

    Returns:
    tuple: (fig, ax) Matplotlib figure and axes objects.
    """
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, alpha=0.6, label=label)
    if show:
        plt.show()
    return fig, ax

def surfPlot(X, Y, Z, label=None, show=True, fig=None, ax=None):
    """
    Create a 3D surface plot.

    Parameters:
    X (ndarray): X coordinates.
    Y (ndarray): Y coordinates.
    Z (ndarray): Z coordinates.
    label (str, optional): Label for the plot.
    show (bool, optional): Whether to show the plot immediately.
    fig (Figure, optional): Matplotlib figure object.
    ax (Axes3D, optional): Matplotlib 3D axes object.

    Returns:
    tuple: (fig, ax) Matplotlib figure and axes objects.
    """
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.spring, alpha=0.9,
                           linewidth=0, antialiased=False, label=label)
    if show:
        plt.show()
    return fig, ax

def create_data(N, order, noise_eps=0.01):
    """
    Create synthetic data using the Franke function.

    Parameters:
    N (int): Number of data points.
    order (int): Polynomial order for the design matrix.
    noise_eps (float, optional): Standard deviation of Gaussian noise added to the data.

    Returns:
    tuple: (X, Y, Z, D) Meshgrid coordinates, Franke function values, and design matrix.
    """
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    X, Y = np.meshgrid(x, y)
    Z = FrankeFunction(X, Y)
    
    # Add white Gaussian noise to the surface
    Z = Z + np.random.normal(0, 1, size=Z.shape) * noise_eps
    D = create_X(X, Y, order=order)
    
    return X, Y, Z, D

def create_terrain_data(filename='datasets/SRTM_data_Norway_1.tif', N=500, down_sample=5, m=5):
    """
    Create terrain data from a given file.

    Parameters:
    filename (str, optional): Path to the terrain data file.
    N (int, optional): Number of data points.
    down_sample (int, optional): Down-sampling factor.
    m (int, optional): Polynomial order for the design matrix.

    Returns:
    tuple: (X, Y, terrain, D) Meshgrid coordinates, terrain data, and design matrix.
    """

    terrain = imageio.imread(filename)
    terrain = terrain[:N, :N]  # change to right size NxN
    terrain = terrain[::down_sample, ::down_sample] # downsample
    x = np.linspace(0, 1, np.shape(terrain)[0])
    y = np.linspace(0, 1, np.shape(terrain)[1])
    X, Y = np.meshgrid(x, y)
    
    D = create_X(X, Y, m)
    return X, Y, terrain, D

def OLS(D_train, D_test, Z_train):
    """
    Perform Ordinary Least Squares (OLS) regression.

    Parameters:
    D_train (ndarray): Design matrix for training data.
    D_test (ndarray): Design matrix for test data.
    Z_train (ndarray): Target values for training data.

    Returns:
    tuple: (beta, pred_train, pred_test)
        - beta (ndarray): Estimated regression coefficients.
        - pred_train (ndarray): Predicted values for training data.
        - pred_test (ndarray): Predicted values for test data.
    """
    D_train_inv = np.linalg.pinv(D_train.T @ D_train)
    beta = D_train_inv @ D_train.T @ Z_train
    pred_train = D_train @ beta
    pred_test = D_test @ beta
    return beta, pred_train, pred_test

def RidgeRegression(D_train, D_test, Z_train, lmb=0.01):
    """
    Perform Ridge Regression on the given training data and predict on both training and test data.

    Parameters:
    D_train (ndarray): Design matrix for training data.
    D_test (ndarray): Design matrix for test data.
    Z_train (ndarray): Target values for training data.
    lmb (float, optional): Regularization parameter (default is 0.01).

    Returns:
    tuple: (beta, pred_train, pred_test)
        - beta (ndarray): Estimated regression coefficients.
        - pred_train (ndarray): Predicted values for training data.
        - pred_test (ndarray): Predicted values for test data.
    """
    n_features = D_train.shape[1]
    I = np.eye(n_features)
    ridge_term = lmb * I
    des_mat_inv = np.linalg.pinv(D_train.T @ D_train + ridge_term)  # Using SVD for numerical stability
    beta = des_mat_inv @ D_train.T @ Z_train
    
    # Predict on train and test data
    pred_train = D_train @ beta
    pred_test = D_test @ beta
    
    return beta, pred_train, pred_test

def LassoRegression(D_train, D_test, Z_train, lmb=0.01):
    """
    Perform Lasso Regression on the given training data and predict on both training and test data.

    Parameters:
    D_train (ndarray): Design matrix for training data.
    D_test (ndarray): Design matrix for test data.
    Z_train (ndarray): Target values for training data.
    lmb (float, optional): Regularization parameter (default is 0.01).

    Returns:
    tuple: (coef, pred_train, pred_test)
        - coef (ndarray): Estimated regression coefficients.
        - pred_train (ndarray): Predicted values for training data.
        - pred_test (ndarray): Predicted values for test data.
    """
    lasso = Lasso(alpha=lmb, fit_intercept=False)
    lasso.fit(D_train, Z_train)
    
    pred_train = lasso.predict(D_train)
    pred_test = lasso.predict(D_test)
    return lasso.coef_, pred_train, pred_test

def split_data(D, Z, test_size=0.2, scaled=False):
    """
    Split data into training and test sets.

    Parameters:
    D (ndarray): Design matrix.
    Z (ndarray): Target values.
    test_size (float, optional): Proportion of the dataset to include in the test split.
    scaled (bool, optional): Whether to scale the data.

    Returns:
    tuple: (D_train, D_test, Z_train, Z_test)
        - D_train (ndarray): Training design matrix.
        - D_test (ndarray): Test design matrix.
        - Z_train (ndarray): Training target values.
        - Z_test (ndarray): Test target values.
    """
    D_train, D_test, Z_train, Z_test = train_test_split(D, Z.ravel(), test_size=test_size, random_state=0)

    if scaled: 
        scaler_Z = StandardScaler(with_std=True)
        scaler_D = StandardScaler(with_std=True)
        Z_train = scaler_Z.fit_transform(Z_train.reshape(-1, 1))
        Z_test = scaler_Z.transform(Z_test.reshape(-1, 1))
        D_train = scaler_D.fit_transform(D_train)
        D_test = scaler_D.transform(D_test)
        
    return D_train, D_test, Z_train, Z_test

def bootstrap_method(D_train, D_test, Z_train, Z_test, n_bootstraps, n_samples, model_type='OLS', lmb=0.01):
    if model_type == 'OLS':
        model = OLS
    elif model_type == 'Ridge':
        model = RidgeRegression
    elif model_type == 'Lasso':
        model = LassoRegression
    else:
        raise ValueError("Invalid model_type. Choose from 'OLS', 'Ridge', or 'Lasso'.")
    Z_preds = np.zeros((Z_test.shape[0], n_bootstraps))
    for boot in range(n_bootstraps):
        D_, Z_ = resample(D_train, Z_train, n_samples=n_samples)
        _, _, pred_test = model(D_, D_test, Z_)
        Z_preds[:,boot] = pred_test.ravel()
    error = np.mean(np.mean((Z_test - Z_preds)**2, axis=1, keepdims=True))
    bias = np.mean((Z_test - np.mean(Z_preds, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(Z_preds, axis=1, keepdims=True))
    return error, bias, variance

def bootstrap_OLS_degrees(N, n_bootstraps, n_samples, degrees, noise_eps=0.01, filename=None, down_sample=5):
    """Implementation of the bootstrap technique"""
    # Create data
    if filename:
        X, Y, Z, _ = create_terrain_data(filename, N=N, down_sample=down_sample, m=1)
    else:
        X, Y, Z, _ = create_data(N=N, order=1, noise_eps=noise_eps)

    
    errors = np.zeros(len(degrees))
    biases = np.zeros(len(degrees))
    variances = np.zeros(len(degrees))
    
    for i, deg in enumerate(degrees):
        # Create design matrix
        D = create_X(X, Y, order=deg)
        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        
        Z_preds = np.zeros((Z_test.shape[0], n_bootstraps))
        # run bootstrap
        for boot in range(n_bootstraps):
            # resample data
            D_, Z_ = resample(D_train, Z_train, n_samples=n_samples)
            _, _, pred_test = OLS(D_, D_test, Z_)
            Z_preds[:,boot] = pred_test.ravel()
            
        biases[i] = np.mean((Z_test - np.mean(Z_preds, axis=1, keepdims = True))**2)
        variances[i] = np.mean(np.var(Z_preds, axis=1, keepdims=True))
        errors[i] = np.mean(np.mean((Z_test - Z_preds)**2, axis=1, keepdims=True))
        
    return errors, biases, variances


def bootstrap_OLS_npoints(n_points, n_bootstraps, n_samples, order, noise_eps=0.01, filename=None, down_sample=5):
    """Implementation of the bootstrap technique"""
    
    errors = np.zeros(len(n_points))
    biases = np.zeros(len(n_points))
    variances = np.zeros(len(n_points))
    
    for i, N in enumerate(n_points):
        # Create data
        # X, Y, Z, D = create_terrain_data(filename, N=N, down_sample=down_sample, m=order)
        X, Y, Z, D = create_data(N=N, order=order, noise_eps=noise_eps)
        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        
        Z_preds = np.zeros((Z_test.shape[0], n_bootstraps))
        # run bootstrap
        for boot in range(n_bootstraps):
            # resample data
            D_, Z_ = resample(D_train, Z_train, n_samples=n_samples)
            _, _, pred_test = OLS(D_, D_test, Z_)
            Z_preds[:,boot] = pred_test.ravel()
            
        biases[i] = np.mean((Z_test - np.mean(Z_preds, axis=1, keepdims = True))**2)
        variances[i] = np.mean(np.var(Z_preds, axis=1, keepdims=True))
        errors[i] = np.mean(np.mean((Z_test - Z_preds)**2, axis=1, keepdims=True))
        
    return errors, biases, variances

def bootstrap_OLS_npoints_terrain(N, down_samples, n_bootstraps, n_samples, order, filename=None):
    """Implementation of the bootstrap technique"""
    
    errors = np.zeros(len(down_samples))
    biases = np.zeros(len(down_samples))
    variances = np.zeros(len(down_samples))
    
    for i, down_sample in enumerate(down_samples):
        # Create data
        # X, Y, Z, D = create_terrain_data(filename, N=N, down_sample=down_sample, m=order)
        X, Y, Z, D = create_terrain_data(filename, N=N, down_sample=down_sample, m=order)
        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        
        Z_preds = np.zeros((Z_test.shape[0], n_bootstraps))
        # run bootstrap
        for boot in range(n_bootstraps):
            # resample data
            D_, Z_ = resample(D_train, Z_train, n_samples=n_samples)
            _, _, pred_test = OLS(D_, D_test, Z_)
            Z_preds[:,boot] = pred_test.ravel()
            
        biases[i] = np.mean((Z_test - np.mean(Z_preds, axis=1, keepdims = True))**2)
        variances[i] = np.mean(np.var(Z_preds, axis=1, keepdims=True))
        errors[i] = np.mean(np.mean((Z_test - Z_preds)**2, axis=1, keepdims=True))
        
    return errors, biases, variances


def sklearn_kfold(Z, D, model_type='OLS', lmb=0.01, n_splits=10):
    if model_type == 'OLS':
        model = LinearRegression()
    elif model_type == 'Ridge':
        model = Ridge(alpha=lmb)
    elif model_type == 'Lasso':
        model = Lasso(alpha=lmb, max_iter=10000)
    else:
        raise ValueError("Invalid model_type. Choose from 'OLS', 'Ridge', or 'Lasso'.")
    
    pipe = make_pipeline(StandardScaler(), model)

    scaler_Z = StandardScaler()
    Z = scaler_Z.fit_transform(Z.ravel().reshape(-1, 1))

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    mse_scores = []
    
    for train, test in kfold.split(D):
        pipe.fit(D[train], Z.ravel()[train])
        Z_pred = pipe.predict(D[test])
        mse = mean_squared_error(Z.ravel()[test], Z_pred)
        mse_scores.append(mse)
    
    mean_mse = np.mean(mse_scores)
    
    return mean_mse


def scores_deg(method, N, max_degree, noise_eps=0.1, filename=None, down_sample=None):
    "Plot MSE and R2 scores for a given method as a function of polynomial degree"
    mse_list = []
    r2_list = []
    if filename:
        X, Y, Z, _ = create_terrain_data(filename=filename, N=N, down_sample=down_sample, m=1)
    else:
        X, Y, Z, _ = create_data(N=N, order=1, noise_eps=noise_eps)
    for degree in range(1, max_degree):
        D = create_X(X, Y, order=degree)

        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        beta, pred_train, pred_test = method(D_train, D_test, Z_train)
        mse_train = mean_squared_error(Z_train, pred_train)
        mse_test = mean_squared_error(Z_test, pred_test)
        r2_train = r2_score(Z_train, pred_train)
        r2_test = r2_score(Z_test, pred_test)
        mse_list.append((mse_train, mse_test))
        r2_list.append((r2_train, r2_test))
    return mse_list, r2_list
    
    
def scores_lambda(method, N, order, lambdas, noise_eps=0.1, filename=None, down_sample=None):
    "Plot MSE and R2 scores for a given method as a function of lambda"
    mse_list = []
    r2_list = []
    if filename:
        X, Y, Z, D = create_terrain_data(filename=filename, N=N, down_sample=down_sample, m=order)
    else:
        X, Y, Z, D = create_data(N, order, noise_eps=noise_eps)
    D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
    for lmb in lambdas:
        beta, pred_train, pred_test = method(D_train, D_test, Z_train, lmb=lmb)
        mse_train = mean_squared_error(Z_train, pred_train)
        mse_test = mean_squared_error(Z_test, pred_test)
        r2_train = r2_score(Z_train, pred_train)
        r2_test = r2_score(Z_test, pred_test)
        mse_list.append((mse_train, mse_test))
        r2_list.append((r2_train, r2_test))
        
    return mse_list, r2_list

def polydeg_lmb_grid_seach(model_type, N, down_sample, noise_eps, lmb_n, lmb_min, lmb_max, mindeg=1, maxdeg=5, k_folds=False, terrain_data=False, show=True, save=False):
    '''
    reg_func is the function for the regression method, either ridge or lasso 
    if k_folds is set to 10 it should perform 10 folds

    '''
    degrees = np.arange(mindeg, maxdeg + 1)
    lambdas = np.logspace(lmb_min, lmb_max, lmb_n)

    # make grid for plotting later
    degrees, lambdas = np.meshgrid(degrees,lambdas)
    mse = np.zeros(np.shape(degrees))

    if not terrain_data:
        data = 'franke'
        print('Using Franke function data')
        for i in range(len(degrees[0])):
            print(f'Processing model degree: {degrees[0, i]}')
            for j in range(len(lambdas[:,0])):
                X, Y, Z, D = create_data(N=N, order=degrees[0, i], noise_eps=noise_eps)

                if k_folds==False:
                    D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
                    beta, pred_train, pred_test = model_type(D_train, D_test, Z_train, lambdas[j, 0])
                    mse_test = mean_squared_error(Z_test, pred_test)
                else:
                    # mse_train, mse_tests = kfold(func, Z, D, k_folds, lambdas[j, 0])
                    # mse_test = np.mean(mse_tests)
                    mse_test = sklearn_kfold(Z, D, model_type=model_type, lmb=lambdas[j, 0], n_splits=10)

                mse[j,i] = mse_test

    else:
        data = 'terrain'
        print('Using digital terrain data')
        # Load the terrain
        filename = '../datasets/SRTM_data_Norway_1.tif'
        # filename = 'datasets/SRTM_data_Norway_2.tif'

        for i in range(len(degrees[0])):
            print(f'Processing model degree: {degrees[0, i]}')
            for j in range(len(lambdas[:,0])):
                X, Y, Z, D = create_terrain_data(filename, N, down_sample, degrees[0, i])
                # print(f"Z min: {np.min(Z)}, Z max: {np.max(Z)}")

                if k_folds==False:
                    D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
                    beta, pred_train, pred_test = model_type(D_train, D_test, Z_train, lambdas[j, 0])
                    mse_test = mean_squared_error(Z_test, pred_test)
                else:
                    # mse_train, mse_tests = kfold(func, Z, D, k_folds, lambdas[j, 0])
                    # mse_test = np.mean(mse_tests)
                     mse_test = sklearn_kfold(Z, D, model_type=model_type, lmb=lambdas[j, 0], n_splits=10)
                mse[j,i] = mse_test

    optimal = np.unravel_index(np.argmin(mse), mse.shape)
    if show and not save:
        plot_mse_heatmap(mse, degrees, lambdas, optimal, save)

    if save:
        try:
            func_str = model_type.__name__
        except:    
            func_str = model_type
        save = data + '_' + func_str + '_lmb_' + str(lmb_min) + '_' + str(lmb_max) + '_poly_' + str(mindeg) + '_' + str(maxdeg)
        if k_folds != False:
            save += '_' + str(k_folds) + 'k_fold_' + 'N' + str(N)

        plot_mse_heatmap(mse, degrees, lambdas, optimal, save)

def scores_deg_terrain(method, N, down_sample, max_degree):
    "Plot MSE and R2 scores for a given method as a function of polynomial degree"
    mse_list = []
    r2_list = []
    for degree in range(1, max_degree):
        filename = '../datasets/SRTM_data_Norway_1.tif'
        X, Y, Z, D = create_terrain_data(filename=filename, N=N, down_sample=down_sample, m=degree)
        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        beta, pred_train, pred_test = method(D_train, D_test, Z_train)
        mse_train = mean_squared_error(Z_train, pred_train)
        mse_test = mean_squared_error(Z_test, pred_test)
        r2_train = r2_score(Z_train, pred_train)
        r2_test = r2_score(Z_test, pred_test)
        mse_list.append((mse_train, mse_test))
        r2_list.append((r2_train, r2_test))
    return mse_list, r2_list
    
    
def scores_lambda_terrain(method, N, down_sample, order, lambdas):
    "Plot MSE and R2 scores for a given method as a function of lambda"
    mse_list = []
    r2_list = []
    filename = '../datasets/SRTM_data_Norway_1.tif'
    X, Y, Z, D = create_terrain_data(filename=filename, N=N, down_sample=down_sample, m=order)
    D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
    for lmb in lambdas:
        beta, pred_train, pred_test = method(D_train, D_test, Z_train, lmb=lmb)
        mse_train = mean_squared_error(Z_train, pred_train)
        mse_test = mean_squared_error(Z_test, pred_test)
        r2_train = r2_score(Z_train, pred_train)
        r2_test = r2_score(Z_test, pred_test)
        mse_list.append((mse_train, mse_test))
        r2_list.append((r2_train, r2_test))
        
    return mse_list, r2_list


