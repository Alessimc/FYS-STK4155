import kagglehub
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd
from utils import  format_param_grid


# Download latest version
path = kagglehub.dataset_download("fmena14/volcanoesvenus")
print("Path to dataset files:", path)

train_dir = os.path.join(path, 'volcanoes_train')
train_images_path = os.path.join(train_dir, 'train_images.csv')
train_labels_path = os.path.join(train_dir, 'train_labels.csv')

train_images = pd.read_csv(train_images_path, header=None)
train_labels = pd.read_csv(train_labels_path)

train_images_scaled = train_images.values / 255 # normalize pixel values to [0,1]

X_train, y_train = train_images_scaled, train_labels['Volcano?']

from sklearn.ensemble import GradientBoostingClassifier

param_space = {
   'n_estimators' : [75, 100, 125, 150],
    'max_depth' : [3, 4, 5, 6],
    'learning_rate' : [0.3, 0.2, 0.1, 0.05],
}

gb_model = GradientBoostingClassifier(random_state=2024)

grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_space,
                               scoring='accuracy', cv=5, verbose=3, n_jobs=-1)

grid_search_gb.fit(X_train, y_train)

# Save data
results_gb = pd.DataFrame(grid_search_gb.cv_results_)
results_gb.to_csv(f"gb_grid_search_{format_param_grid(param_space)}.csv", index=False)

print("Best hyperparameters:", grid_search_gb.best_params_)

# Use the best model to make predictions
best_gb_model = grid_search_gb.best_estimator_
best_accuracy_gb = grid_search_gb.best_score_
print(best_accuracy_gb)

print(f"Tuned Gradient Boosting Accuracy: {best_accuracy_gb * 100:.2f}")
dump(best_gb_model, f'best_gb_model_val_acc_{best_accuracy_gb * 100:.2f}.joblib')
