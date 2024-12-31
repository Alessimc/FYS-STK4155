import kagglehub
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
import pandas as pd
from utils import  format_param_grid


# Download latest version
path = kagglehub.dataset_download("fmena14/volcanoesvenus")
print("Path to dataset files:", path)

# List files in the dataset directory
files = os.listdir(path)
print("Files in the dataset directory:", files)

# Assign paths to train and test directories
train_dir = os.path.join(path, 'volcanoes_train')

train_dir = os.path.join(path, 'volcanoes_train')
train_images_path = os.path.join(train_dir, 'train_images.csv')
train_labels_path = os.path.join(train_dir, 'train_labels.csv')

train_images = pd.read_csv(train_images_path, header=None)
train_labels = pd.read_csv(train_labels_path)

train_images_scaled = train_images.values / 255 # normalize pixel values to [0,1]

X_train, y_train = train_images_scaled, train_labels['Volcano?']

import xgboost as xgb

# param_space = {
#    'n_estimators' : [100, 125, 150],
#     'max_depth' : [4, 5, 6],
#     'learning_rate' : [0.2, 0.1, 0.05],
# }

param_space = {
        #'min_child_weight': [1, 5, 10],
        #'gamma': [0, 0.5, 1],
        #'subsample': [0.6, 0.8, 1.0],
        #'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators' : [125, 150, 175],
        'max_depth': [3,4,5],
        'learning_rate' : [0.3, 0.2, 0.1]
        }

xgb_model = xgb.XGBClassifier(eval_metric='logloss', objective='binary:logistic')

xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_train)
print("Train Accuracy:", accuracy_score(y_train, predictions))

grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_space,
                               scoring='accuracy', cv=5, verbose=3, n_jobs=1)

grid_search_xgb.fit(X_train, y_train)

# Save data
results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)
results_xgb.to_csv(f"xgb_grid_search_{format_param_grid(param_space)}.csv", index=False)

print("Best hyperparameters:", grid_search_xgb.best_params_)

# Use the best model to make predictions
best_xgb_model = grid_search_xgb.best_estimator_
best_accuracy_xgb = grid_search_xgb.best_score_
print(best_accuracy_xgb)
# Evaluate accuracy

print(f"Tuned Gradient Boosting Accuracy: {best_accuracy_xgb * 100:.2f}")
dump(best_xgb_model, f'best_xgb_model_val_acc_{best_accuracy_xgb * 100:.2f}.joblib')
