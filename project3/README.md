# Seeing the Volcanoes for the Trees

<img src="https://upload.wikimedia.org/wikipedia/commons/0/0e/Venus_Rotation_Movie.gif" width="250">

This is repository contains code for the study of tree based methods in machine learning for classifying volcanoes on Venus. A comparison of accuracy and AUC for seven different methods have been conducted. The results emphasize the robustness of the ensemble methods. The methods that have been studied are:
- Decision trees
- Bagging
- Random forest
- Gradient Boosting
- Extreme Gradient Boosting (XGBoost)
- Feed Forward Neural Network
- Logistic Regression

## Project Structure
- **[./notebooks](./notebooks/)** - Directory containing various python scripts and jupyter notebooks for performing the gridsearches and optimization of the different models.
- **[./models](./models/)** - Optimized models saved after gridseach and training.
- **[./grid_search_data](./grid_search_data/)** - Various `.csv` containing the gridsearch data for all models. 
- **[./figures](./figures/)** - Directory containing figures use in the report. 
- **[./roc_curves](./roc_curves/)** - Directory containing `.csv` files used for generating the ROC curves for the different models. 
- **[plot_tuning.ipynb](./plot_tuning.ipynb)** - Jupyter notebook for visualizing the results.
- **[README.md](./README.md)** - This README file.

