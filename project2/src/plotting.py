# File that constins all helper plotting functions
import matplotlib.pyplot as plt
import seaborn as sns

def make_plot(type):
    # Close previous figures to prevent blank plots in Jupyter
    plt.close('all')
    plt.style.use('ggplot')

    # Set standardized params for each type of plot
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.labelspacing'] = 0.25    
    if type == 'half-width':
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13

        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        return fig, ax
    elif type == 'full-width':
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11

        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        return fig, ax
    elif type == 'standard':
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9

        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        return fig, ax
    elif type == 'standard-2':
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13

        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        return fig, ax
    elif type == 'gridsearch':
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18

        fig, ax = plt.subplots(figsize = (4.2, 3.2))
        return fig, ax
    elif type == 'gridsearch_regression':
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        fig, ax = plt.subplots(figsize = (4.2, 2.8))
        return fig, ax
    elif type == 'normal-plot':
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['xtick.labelsize'] = 18
        fig, ax = plt.subplots(figsize = (4.4, 3.))
        return fig, ax
    else:
        print('Invalid plot type - change make_plot function call')
    
def plot_heatmap(scores, xticks, yticks, figname=None):
    _, fig = make_plot('gridsearch')
    sns.heatmap(scores, annot=True, xticklabels=xticks, yticklabels=yticks, cbar_kws={'label': 'Accuracy'}, cmap = 'RdYlGn', vmin=0.67, vmax=1, ax=fig)
    plt.xlabel('Number of layers')
    plt.ylabel('Number of nodes')
    plt.grid(False)
    if figname != None:
        plt.savefig(f'../figures/{figname}.pdf', bbox_inches='tight')
    plt.show()