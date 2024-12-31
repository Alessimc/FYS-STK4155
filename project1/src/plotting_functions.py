"""
This file contains all functions used for plotting results
"""

# imports
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
#import imageio.v2 as imageio

#from sklearn.linear_model import Lasso
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.utils import resample


# Suppress ConvergenceWarning and LinAlgWarning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)  # This covers LinAlg warnings

import general_functions
import importlib
importlib.reload(general_functions)
from general_functions import *

import scienceplots
# plt.style.use(['science', 'ieee'])
# plt.rcParams.update({'figure.dpi': '100'})
plt.style.use('ggplot')

def make_plot(nrows, ncols, full_width=False):
    # make standardised params for each type of plot
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    if full_width:
        plt.rcParams['font.size'] = 10  # Slightly larger font size for overall text
        plt.rcParams['axes.labelsize'] = 10  # Font size for axis labels
        plt.rcParams['xtick.labelsize'] = 9   # Font size for x-axis tick labels
        plt.rcParams['ytick.labelsize'] = 9   # Font size for y-axis tick labels

        # Adjusting the figure size for single column (e.g., ~4.2 inches wide)
        fig, ax = plt.subplots(figsize=(4.2, 3.2))  # Adjust height as needed
        return fig, ax

    if nrows == 2 and ncols == 1:
        plt.rcParams['font.size'] = 11  # or 9 for slightly larger fonts
        plt.rcParams['axes.labelsize'] = 11  # font size for axis labels
        plt.rcParams['xtick.labelsize'] = 10  # font size for x-axis tick labels
        plt.rcParams['ytick.labelsize'] = 10  # font size for y-axis tick labels

        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        fig, ax = plt.subplots(figsize=(5, 2))
    elif nrows == 1 and ncols == 1:
        plt.rcParams['font.size'] = 9  # or 9 for slightly larger fonts
        plt.rcParams['axes.labelsize'] = 9  # font size for axis labels
        plt.rcParams['xtick.labelsize'] = 8  # font size for x-axis tick labels
        plt.rcParams['ytick.labelsize'] = 8  # font size for y-axis tick labels
        fig, ax = plt.subplots(figsize=(3.5, 2.625))
    elif nrows == 1 and ncols == 3:
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        fig, ax = plt.subplots(figsize=(3.5, 2.625))
    elif nrows == 2 and ncols == 2:
        plt.rcParams['font.size'] = 8  # Slightly larger font size for overall text
        plt.rcParams['axes.labelsize'] = 8  # Font size for axis labels
        plt.rcParams['xtick.labelsize'] = 7   # Font size for x-axis tick labels
        plt.rcParams['ytick.labelsize'] = 7   # Font size for y-axis tick labels

        # Adjusting the figure size for single column (e.g., ~4.2 inches wide)
        fig, ax = plt.subplots(figsize=(4.2, 3.2))  # Adjust height as needed
        return fig, ax
    else: 
        raise ValueError('nrows and ncols not supported')
    return fig, ax 

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
        fig = plt.figure(figsize=(4.2, 3.2))
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, color='blue', alpha=0.5, label=label)
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
        fig = plt.figure(figsize=(4.2, 3.2))
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.spring, alpha=0.8,
                           linewidth=0, antialiased=False, label=label)
    if show:
        plt.show()
    return fig, ax        

def plot_betas_OLS_for_polydeg(N = 50, max_degree = 5, savefile=None):
    poly_order = np.arange(1,max_degree+1)
    betas = []
    for degree in range(1, max_degree+1):
        X, Y, Z, D = create_data(N, order=degree, noise_eps=0.01)
        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        beta, pred_train, pred_test = OLS(D_train, D_test, Z_train)
        betas.append(beta)
        
    fig, ax = make_plot(1, 3)
    for order in range(0, max_degree): # actual poly order is order + 1
        order_vals = np.full(len(betas[order]), poly_order[order])
        plt.scatter(betas[order], order_vals)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Polynomial order')
    plt.xlabel(r'$\beta_i$ values')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, bbox_inches='tight')
    plt.show()

def plot_betas_OLS_for_polydeg_terrain(N = 500, down_sample=5, max_degree = 5, savefile=None, filename=None):
    poly_order = np.arange(1,max_degree+1)
    betas = []
    for degree in range(1, max_degree+1):
        X, Y, Z, D = create_terrain_data(filename, N, down_sample, m=degree)
        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        beta, pred_train, pred_test = OLS(D_train, D_test, Z_train)
        betas.append(beta)
        
    fig, ax = make_plot(1, 1)
    for order in range(0, max_degree): # actual poly order is order + 1
        order_vals = np.full(len(betas[order]), poly_order[order])
        plt.scatter(betas[order], order_vals)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Polynomial order')
    plt.xlabel(r'$\beta_i$ values')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, bbox_inches='tight')
    plt.show()

def plot_betas_for_lmb(func, N, order, lambdas, savefile=None):
    X, Y, Z, D = create_data(N, order, noise_eps=0.01)
    D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
    
    betas = []
    for lmb in lambdas:
        beta, pred_train, pred_test = func(D_train, D_test, Z_train, lmb=lmb)
        betas.append(beta)

    fig, ax = make_plot(1, 3)
    for i in range(len(lambdas)):
        lmb_vals = np.full(len(betas[order]), lambdas[i])
        plt.scatter(betas[i], lmb_vals)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yscale('log')
    plt.ylabel(r'$\lambda$')
    plt.xlabel(r'$\beta_i$ values')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, bbox_inches='tight')

def plot_betas_for_lmb_terrain(func, N, down_sample, order, lambdas, savefile=None, filename=None):
    X, Y, Z, D = create_terrain_data(filename, N, down_sample, m=order)
    D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
    
    betas = []
    for lmb in lambdas:
        beta, pred_train, pred_test = func(D_train, D_test, Z_train, lmb=lmb)
        betas.append(beta)

    fig, ax = make_plot(1, 1)
    for i in range(len(lambdas)):
        lmb_vals = np.full(len(betas[order]), lambdas[i])
        plt.scatter(betas[i], lmb_vals)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yscale('log')
    plt.ylabel(r'$\lambda$')
    plt.xlabel(r'$\beta_i$ values')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, bbox_inches='tight')

def plot_bias_variance_tradeoff_ols(iterable, errors, biases, variances,x_label='', show=True, filename=None, logx=False):
    """Plot the bias-variance tradeoff for the OLS model"""
    # plot mse as function of degree
    fig, ax = make_plot(1, 1)
    plt.semilogy(iterable, errors, label='MSE')
    plt.semilogy(iterable, biases, label='Bias')
    plt.semilogy(iterable, variances, label='Variance')
    if logx:
        plt.xscale('log')

    plt.xlabel(x_label)
    plt.ylabel('Value')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()

def plot_MSE_boot_kfold(N, max_order, n_bootstraps, n_samples, n_splits, noise_eps=0.01, lmb=0.01, model_type='OLS', savefile=None, filename=None, down_sample=None):
    """Generates a plot of the MSE using bootstrap and kfold for degrees up to max_order"""
    # Generate data
    if filename:
        X, Y, Z, _ = create_terrain_data(filename, N, down_sample, m=1)
    else:
        X, Y, Z, _ = create_data(N, max_order, noise_eps=noise_eps)
    
    mse_boot = np.zeros(max_order)
    mse_kfold = np.zeros(max_order)
    
    for order in range(1, max_order+1):
        D = create_X(X, Y, order)
        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        mse_boot[order-1], _, _ = bootstrap_method(D_train, D_test, Z_train, Z_test, n_bootstraps, n_samples, model_type=model_type, lmb=lmb)
        mse_kfold[order-1] = sklearn_kfold(Z, D, model_type=model_type, lmb=0.01, n_splits=n_splits)
        
    fig, ax = make_plot(1, 1)
    plt.plot(range(1, max_order+1), mse_boot, label='Bootstrap')
    plt.plot(range(1, max_order+1), mse_kfold, label='KFold')
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE')
    plt.legend()
    if savefile:
        plt.savefig(savefile, bbox_inches='tight')
    plt.show()

def plot_MSE_boot_kfold_terrain(N, max_order, n_bootstraps, n_samples, n_splits, lmb=0.01, model_type='OLS', savefile=None, filename='../datasets/SRTM_data_Norway_1.tif', down_sample=None):
    """Generates a plot of the MSE using bootstrap and kfold for degrees up to max_order"""
    # Generate data
    X, Y, Z, _ = create_terrain_data(filename, N, down_sample, m=1)
    
    mse_boot = np.zeros(max_order)
    mse_kfold = np.zeros(max_order)
    
    for order in range(1, max_order+1):
        D = create_X(X, Y, order)
        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        mse_boot[order-1], _, _ = bootstrap_method(D_train, D_test, Z_train, Z_test, n_bootstraps, n_samples, model_type=model_type, lmb=lmb)
        mse_kfold[order-1] = sklearn_kfold(Z, D, model_type=model_type, lmb=0.01, n_splits=n_splits)
        
    fig, ax = make_plot(1, 1)
    plt.plot(range(1, max_order+1), mse_boot, label='Bootstrap')
    plt.plot(range(1, max_order+1), mse_kfold, label='KFold')
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE')
    plt.legend()
    if savefile:
        plt.savefig(savefile, bbox_inches='tight')
    plt.show()

def plot_mse_heatmap(mse, degrees, lambdas, optimal, save=False):
    
    fig, ax = make_plot(1, 1, full_width=True)
    plt.contourf(degrees, lambdas, mse, 50, cmap="RdYlGn_r")
    plt.colorbar(label='$MSE$')
    plt.ylabel(r'$\lambda$')
    plt.yscale("log")
    plt.scatter(degrees[optimal], lambdas[optimal], marker="x", s=150, label=f'MSE: {mse[optimal]:.3f}, \nLambda: {lambdas[optimal]:.2e}, \nDegree: {degrees[optimal]}') 
    plt.legend(framealpha=0.5)

    plt.xlabel('Polynomial Degree')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(r'$\lambda$')
    
    if save != False:
        plt.savefig(f'../figures/{save}.pdf', bbox_inches='tight')
    # Show plot
    plt.show()


def plot_OLS_deg_terrain(N, down_sample, noise_eps, mindeg=1, maxdeg=5, k_folds=False, show=True, save=False):
    
    degrees = np.arange(mindeg, maxdeg + 1)
    mse = np.zeros(len(degrees))
    filename = 'datasets/SRTM_data_Norway_1.tif'
    N = 1000
    for i in range(len(degrees)):
        print(f'Processing model degree: {degrees[i]}')

        X, Y, Z, D = create_terrain_data(filename, N, down_sample, degrees[i])
        D_train, D_test, Z_train, Z_test = split_data(D, Z, test_size=0.2, scaled=True)
        beta, pred_train, pred_test = OLS(D_train, D_test, Z_train)

        if k_folds==False:
            mse_test = mean_squared_error(Z_test, pred_test)
        else:
            mse_train, mse_tests = kfold(OLS, Z, D, k_folds)
            mse_test = np.mean(mse_tests)
        mse[i] = mse_test

    plt.plot(degrees, mse)
    plt.show()

def plot_terrain_for_methsects(N, down_sample):
    """Plot the terrain data for the method sections"""
    #filename = '/Users/sophusbredesengullbekk/Documents/master/høst2024/FYS-STK4155/github/FYS-STK-Proj1/datasets/SRTM_data_Norway_1.tif'
    filename = '../datasets/SRTM_data_Norway_1.tif'
    _, _, Z, _ = create_terrain_data(filename=filename, N=N, down_sample=down_sample, m=1)
    
    fig, ax = make_plot(1, 1)
    plt.imshow(Z, cmap='coolwarm')
    plt.axis('off')
    clp = plt.colorbar()
    clp.set_label('Height [m]')
    plt.savefig('../figures/terrain_data.pdf', bbox_inches='tight')
    plt.show()